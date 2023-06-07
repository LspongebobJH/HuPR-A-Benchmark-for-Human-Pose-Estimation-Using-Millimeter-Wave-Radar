import os
import torch
import numpy as np
import torch.optim as optim
from models import HuPRNet
from misc import plotHumanPose
from datasets.dataset import getDataset, collate_fn
import torch.utils.data as data
import torch.nn.functional as F
from tools.base import BaseRunner
import wandb
import pickle
import omegaconf
import numpy as np

import horovod.torch as hvd

class Runner(BaseRunner):
    def __init__(self, cfg):
        super(Runner, self).__init__(cfg)    
    
    def eval(self, dataSet, dataLoader, visualization=True, epoch=-1):
        self.model.eval()
        loss_list = []
        self.logger.clear(len(dataLoader.dataset))
        savePreds = []
        for idx, (batch, frames_list) in enumerate(dataLoader):
            keypoints = batch['jointsGroup']
            bbox = batch['bbox']
            imageId = batch['imageId']
            with torch.no_grad():
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().cuda()
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().cuda()
                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
                loss, loss2, preds, gts = self.lossComputer.computeLoss(preds, keypoints)
                self.logger.display(loss, loss2, keypoints.size(0), epoch)
                if visualization:
                    plotHumanPose(preds*self.imgHeatmapRatio, self.cfg, 
                                  self.visdir, imageId, None)
                    # # for drawing GT
                    # plotHumanPose(gts*self.imgHeatmapRatio, self.cfg, 
                    #               self.visdir, imageId, None)

            self.saveKeypoints(savePreds, preds*self.imgHeatmapRatio, bbox, imageId)
            loss_list.append(loss.item())
        self.writeKeypoints(savePreds)
        if self.cfg.RUN.keypoints:
            ap = dataSet.evaluateEach(self.dir)
        ap = dataSet.evaluate(self.dir)
        return ap

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            loss_list = []
            loss1_list = []
            loss2_list = []
            self.logger.clear(len(self.trainLoader.dataset))
            for idxBatch, (batch, frames_list) in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                keypoints = batch['jointsGroup']
                bbox = batch['bbox']
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().cuda()
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().cuda()
                video_id = batch['video_id']
                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, video_id)
                loss, loss1, loss2, _, _ = self.lossComputer.computeLoss(preds, keypoints)
                loss.backward()
                self.optimizer.step()                    
                self.logger.display(loss, loss2, keypoints.size(0), epoch)

            if (self.cfg.RUN.use_horovod and hvd.rank() == 0) or not self.cfg.RUN.use_horovod:
                if idxBatch % self.cfg.TRAINING.lrDecayIter == 0: #200 == 0:
                    self.adjustLR(epoch)
                    hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
                loss_list.append(loss.item())
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

                ap = self.eval(self.evalSet, self.evalLoader, visualization=False, epoch=epoch)
                self.run.log({
                    'train/loss_mean': np.mean(loss_list), 
                    'train/cnn_loss_mean': np.mean(loss1_list),
                    'train/gnn_loss_mean': np.mean(loss2_list), 
                    'eval/ap': ap
                })

                self.saveModelWeight(epoch, ap)
                self.saveLosslist(epoch, loss_list, 'train')

    def main(self):
        if self.cfg.RUN.use_horovod:
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())

        wandb_cfg = omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        
        if not self.cfg.RUN.test:
            self.trainSet = getDataset('train', self.cfg)
            if self.cfg.RUN.use_horovod:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainSet, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
                self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batchSize,
                                #   shuffle=self.cfg.DATASET.shuffle,
                                  shuffle=False,
                                  num_workers=self.cfg.SETUP.numWorkers,
                                  collate_fn=collate_fn,
                                  sampler=train_sampler)
            else:
                self.trainLoader = data.DataLoader(self.trainSet,
                                    self.cfg.TRAINING.batchSize,
                                    # shuffle=self.cfg.DATASET.shuffle,
                                    shuffle=False,
                                    num_workers=self.cfg.SETUP.numWorkers,
                                    collate_fn=collate_fn)
            self.evalSet = getDataset('val', self.cfg)
            self.evalLoader = data.DataLoader(self.evalSet, 
                                self.cfg.TEST.batchSize,
                                shuffle=False,
                                num_workers=self.cfg.SETUP.numWorkers, 
                                collate_fn=collate_fn)
            
        else:
            self.trainLoader = [0] # an empty loader
        self.testSet = getDataset('test', self.cfg)
        self.testLoader = data.DataLoader(self.testSet, 
                              self.cfg.TEST.batchSize,
                              shuffle=False,
                              num_workers=self.cfg.SETUP.numWorkers, 
                              collate_fn=collate_fn)
        self.model = HuPRNet(self.cfg).cuda()
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)
        self.initialize(LR)
        self.beta = 0.0
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        if self.cfg.RUN.debug:
            self.cfg.TRAINING.epochs = 2
            self.cfg.RUN.logdir = self.cfg.RUN.visdir = 'test'

        if self.cfg.RUN.use_horovod:
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        run = wandb.init(config=wandb_cfg, project=self.cfg.RUN.project, dir='/mnt/jiahanli/wandb')
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("eval/*", step_metric="epoch")
        self.run = run

        print("=== Start Training ===")
        self.train()
        print("=== End Training ===")

        if (self.cfg.RUN.use_horovod and hvd.rank() == 0) or not self.cfg.RUN.use_horovod:
            print("=== Start Evaluation on Test Set ===")
            ap = self.eval(self.testSet, self.testLoader, visualization=True, epoch=self.cfg.TRAINING.epochs - 1)
            print("=== End Evaluation on Test Set ===")
            self.run.log({'test/ap': ap})

