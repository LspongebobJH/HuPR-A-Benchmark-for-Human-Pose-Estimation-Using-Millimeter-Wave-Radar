import os
import torch
import numpy as np
import torch.optim as optim
from models import HuPRNet, HuPRNetSingle
from misc import plotHumanPose
from datasets.dataset import HuPR3D_horivert
import torch.utils.data as data
import torch.nn.functional as F
from tools.base import BaseRunner
import omegaconf
import numpy as np
from math import ceil
from tqdm import tqdm
import time
from misc.losses import LossComputer

import json
import horovod.torch as hvd

from torch.utils.tensorboard import SummaryWriter
from tools.utils import sanity_check

class Runner(BaseRunner):
    def __init__(self, cfg):
        super(Runner, self).__init__(cfg)    

    def model_forward(self, batch):
        keypoints = batch['jointsGroup']

        if self.cfg.DATASET.direction == 'all':
            VRDAEmaps_hori = batch['VRDAEmap_hori'].float().cuda()
            VRDAEmaps_vert = batch['VRDAEmap_vert'].float().cuda()
            heatmap, gcn_heatmap = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
        else:
            VRDAEmaps = batch['VRDAEmap'].float().cuda()
            heatmap, gcn_heatmap = self.model(VRDAEmaps)
        loss, loss1, loss2, pred2d, gt2d = self.lossComputer.computeLoss(heatmap, gcn_heatmap, keypoints)
        return heatmap, gcn_heatmap, loss, loss1, loss2, pred2d, gt2d
    
    def eval(self, dataSet, dataLoader, visualization=False, epoch=-1):
        self.model.eval()
        loss_list = []
        savePreds = []
        for idx, batch in enumerate(tqdm(dataLoader)):
            self.model.eval()
            _, _, loss, _, _, pred2d, _ = self.model_forward(batch) # pred2d is gcn output heatmap
            
            bbox = batch['bbox']
            imageId = batch['imageId']
            if visualization:
                plotHumanPose(pred2d*self.imgHeatmapRatio, self.cfg, 
                                self.visdir, imageId, None)
                # # for drawing GT
                # plotHumanPose(gts*self.imgHeatmapRatio, self.cfg, 
                #               self.visdir, imageId, None)

            self.saveKeypoints(savePreds, pred2d*self.imgHeatmapRatio, bbox, imageId)
            loss_list.append(loss.item())

            if self.cfg.RUN.debug:
                break
        self.writeKeypoints(savePreds, dataSet.phase)
        if self.cfg.RUN.keypoints:
            ap = dataSet.evaluateEach(self.result_dir)
        else:
            ap = dataSet.evaluate(self.result_dir)
        return ap

    def train(self):
        if hasattr(self, 'start_ap'):
            best_ap = self.start_ap
        else:
            best_ap = -1

        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            loss_list = []
            loss1_list = []
            loss2_list = []

            if sanity_check(check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
                num_batches = len(self.trainLoader)
            else:
                num_batches = ceil(len(self.trainLoader) / hvd.local_size())

            time_st = time.time()
            for idxBatch, batch in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                self.model.train()
                _, _, loss, loss1, loss2, _, _ = self.model_forward(batch)
                loss.backward()
                self.optimizer.step()                    

                if sanity_check(check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
                    loss_list.append(loss.item())
                    loss1_list.append(loss1.item())
                    loss2_list.append(loss2.item())

                    print(f'TRAIN, '
                        f'Epoch: {epoch}/{self.cfg.TRAINING.epochs}, '
                        f'Batch: {idxBatch}/{num_batches}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Loss1: {loss1.item():.4f}, '
                        f'Loss2: {loss2.item():.4f}, '
                        f'Batch time: {time.time() - time_st:.2f}s')
                time_st = time.time()
                    
                if self.cfg.RUN.debug:
                    break

            if sanity_check(check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
                # if idxBatch % self.cfg.TRAINING.lrDecayIter == 0: #200 == 0:
                #     self.adjustLR(epoch)
                #     hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
                
                time_st = time.time()
                ap = self.eval(self.evalSet, self.evalLoader, visualization=False, epoch=epoch)
                print(f'EVAL, '
                    f'Epoch: {epoch}/{self.cfg.TRAINING.epochs}, '
                    f'Ap: {ap:.4f}, '
                    f'Time: {time.time() - time_st:.2f}s')

                if not self.cfg.RUN.debug:
                    self.run.add_scalars('train', {
                        'loss_mean': np.mean(loss_list),
                        'cnn_loss_mean': np.mean(loss1_list),
                        'gnn_loss_mean': np.mean(loss2_list), 
                    }, global_step=epoch)
                    self.run.add_scalar('eval/ap', ap, global_step=epoch)
                
                ap = self.saveStatus(epoch, ap, best_ap, loss_list)
                    
    def main(self):
        if self.cfg.RUN.debug:
            self.cfg.TRAINING.epochs = 3 # TODO: should remove comment
            self.cfg.RUN.logdir = self.cfg.RUN.visdir = 'test'
            self.cfg.RUN.use_horovod = False
            self.cfg.RUN.visualization = False

        if self.cfg.RUN.use_horovod:
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())
            self.rank = hvd.local_rank()
        else:
            torch.cuda.set_device(0)
            self.rank = 0

        if sanity_check(check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
            dir_list = [self.dir, self.visdir, self.checkpoint_dir, self.result_dir, self.tensorboard_dir]
            for _dir in dir_list:
                if not os.path.isdir(_dir):
                    os.mkdir(_dir)
            
        if sanity_check(check_debug=True, debug=self.cfg.RUN.debug, 
                        check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
            cfg = omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
            with open(os.path.join(self.dir, "config.json"), "w") as file:
                json.dump(cfg, file)

        self.lossComputer = LossComputer(self.cfg)

        if self.cfg.DATASET.direction == 'all':
            self.model = HuPRNet(self.cfg).cuda()
        elif self.cfg.DATASET.direction in ['hori', 'vert']:
            self.model = HuPRNetSingle(self.cfg).cuda()

        if not self.cfg.RUN.eval:
            self.trainSet = HuPR3D_horivert('train', self.cfg)
            if self.cfg.RUN.use_horovod:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainSet, num_replicas=hvd.size(), rank=self.rank, shuffle=False)
                self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batchSize,
                                #   shuffle=self.cfg.DATASET.shuffle,
                                  shuffle=False,
                                  num_workers=self.cfg.SETUP.numWorkers,
                                  sampler=train_sampler)
            else:
                self.trainLoader = data.DataLoader(self.trainSet,
                                    self.cfg.TRAINING.batchSize,
                                    # shuffle=self.cfg.DATASET.shuffle,
                                    shuffle=False,
                                    num_workers=self.cfg.SETUP.numWorkers)
            self.evalSet = HuPR3D_horivert('val', self.cfg)
            self.evalLoader = data.DataLoader(self.evalSet, 
                                self.cfg.TEST.batchSize,
                                shuffle=False,
                                num_workers=self.cfg.SETUP.numWorkers)
            
            LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)

            if self.cfg.TRAINING.optimizer == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
            elif self.cfg.TRAINING.optimizer == 'adam':  
                self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)

            if sanity_check(check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank, 
                            check_load_checkpoinit=True, load_checkpoint=self.cfg.RUN.load_checkpoint):
                self.loadModelWeight('checkpoint', checkpoint_dir=self.cfg.RUN.checkpoint_dir, continue_training=True)

            if self.cfg.RUN.use_horovod:
                self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())
                hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

            if sanity_check(check_debug=True, debug=self.cfg.RUN.debug, 
                            check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
                self.run = SummaryWriter(log_dir=self.tensorboard_dir)

            print('==========>Train set size:', len(self.trainLoader))
            print('==========>Eval set size:', len(self.evalLoader))
            
        else:
            self.trainLoader = [0] # an empty loader

        self.testSet = HuPR3D_horivert('test', self.cfg)
        self.testLoader = data.DataLoader(self.testSet, 
                              self.cfg.TEST.batchSize,
                              shuffle=False,
                              num_workers=self.cfg.SETUP.numWorkers)

        self.beta = 0.0
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        
        print('==========>Test set size:', len(self.testLoader))

        if not self.cfg.RUN.eval:
            print("=== Start Training ===")
            self.train()
            print("=== End Training ===")

        if sanity_check(check_horovod=True, use_horovod=self.cfg.RUN.use_horovod, rank=self.rank):
            print("=== Start Evaluation on Test Set ===")
            time_st = time.time()
            self.loadModelWeight('model_best')
            ap = self.eval(self.testSet, self.testLoader, visualization=self.cfg.RUN.visualization, epoch=self.cfg.TRAINING.epochs - 1)
            print("=== End Evaluation on Test Set ===")
            print(f'TEST, '
                f'Ap: {ap:.4f}, '
                f'Time: {time.time() - time_st:.2f}s')
            
            if not self.cfg.RUN.debug:
                self.run.add_scalar('test/ap', ap)
                self.run.flush()
                self.run.close()

        

