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
import ray
from ray.air import session, Checkpoint, CheckpointConfig
from ray import train
import ray.train.torch as rt
from ray.air.config import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import setup_wandb
import pickle
import omegaconf
import numpy as np

class Runner(BaseRunner):
    def __init__(self, cfg):
        super(Runner, self).__init__(cfg)    
        
        if cfg.RUN.debug:
            cfg.TRAINING.epochs = 2
            cfg.RUN.logdir = cfg.RUN.visdir = 'test'

        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)
        self.initialize(LR)
        self.beta = 0.0
    
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
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
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
        best_ap = -1.0
        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            self.model.train()
            loss_list = []
            loss2_list = []
            self.logger.clear(len(self.trainLoader.dataset))
            for idxBatch, (batch, frames_list) in enumerate(self.trainLoader):
                self.optimizer.zero_grad()
                keypoints = batch['jointsGroup']
                bbox = batch['bbox']
                if not self.cfg.RUN.use_ray:
                    VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
                    VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
                video_id = batch['video_id']
                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert, video_id)
                loss, loss2, _, _ = self.lossComputer.computeLoss(preds, keypoints)
                loss.backward()
                self.optimizer.step()                    
                self.logger.display(loss, loss2, keypoints.size(0), epoch)
                if idxBatch % self.cfg.TRAINING.lrDecayIter == 0: #200 == 0:
                  self.adjustLR(epoch)
                loss_list.append(loss.item())
                loss2_list.append(loss2.item())
            ap = self.eval(self.evalSet, self.evalLoader, visualization=False, epoch=epoch)
            self.run.log({
                'train/loss_mean': np.mean(loss_list), 'train/gcn_loss_mean': np.mean(loss2_list), 'eval/ap': ap
            })

            if self.cfg.RUN.use_ray:
                #save best performance model 
                if best_ap < ap:
                    state_dict = self.model.state_dict()
                    checkpoint = Checkpoint.from_dict({
                        "epoch": epoch, "ap": ap, 
                        "config": omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
                        "best_model": state_dict, "optimizer": self.optimizer.state_dict()})
                    session.report({'ap': ap}, checkpoint=checkpoint)
                    print(f"Save best | Epoch {epoch}/{self.cfg.TRAINING.epochs - 1} | ap {ap}")
                    print(f"Save best dir {session.get_trial_dir()}")
                    self.run.summary['best_ap'] = ap
                    self.run.summary['best_epoch'] = epoch
                    best_ap = ap
                else:
                    session.report({'ap': ap})
            else:
                self.saveModelWeight(epoch, ap)
                self.saveLosslist(epoch, loss_list, 'train')

    def main(self):
        wandb_cfg = omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        
        if not self.cfg.RUN.test:
            self.trainSet = getDataset('train', self.cfg)
            self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batchSize,
                                  shuffle=self.cfg.DATASET.shuffle,
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
        self.model = HuPRNet(self.cfg)
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch

        if self.cfg.RUN.use_ray:
            self.model = rt.prepare_model(self.model)
            self.optimizer = rt.prepare_optimizer(self.optimizer)
            self.trainLoader = rt.prepare_data_loader(self.trainLoader)
            run = setup_wandb(wandb_cfg, project=self.cfg.RUN.project, dir='/mnt/jiahanli/wandb')
        else:
            self.model = self.model.to(self.device)
            run = wandb.init(config=wandb_cfg, project=self.cfg.RUN.project, dir='/mnt/jiahanli/wandb')
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("eval/*", step_metric="epoch")
        self.run = run

        print("=== Start Training ===")
        self.train()
        print("=== End Training ===")

        print("=== Start Evaluation on Test Set ===")
        ap = self.eval(self.testSet, self.testLoader, visualization=True, epoch=self.cfg.TRAINING.epochs - 1)
        print("=== End Evaluation on Test Set ===")
        self.run.log({'test/ap': ap})
        return ap

