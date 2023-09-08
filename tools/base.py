import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from misc.logger import Logger
import torch.utils.data as data
import torch.nn.functional as F
from misc.losses import LossComputer

import horovod.torch as hvd

class BaseRunner():
    def __init__(self, cfg):
        self.device = f'cuda:{cfg.RUN.gpu}' if torch.cuda.is_available() else 'cpu'
        np.random.seed(cfg.RUN.seed)
        torch.manual_seed(cfg.RUN.seed)
        torch.cuda.manual_seed_all(cfg.RUN.seed)

        self.dir = './logs/' + cfg.RUN.logdir
        self.visdir = './visualization/' + cfg.RUN.visdir
        self.checkpoint_dir = os.path.join(self.dir, 'checkpoints')
        self.result_dir = os.path.join(self.dir, 'results')
        self.tensorboard_dir = os.path.join(self.dir, 'tensorboard')
        
        self.cfg = cfg
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.dimsWidthHeight = (self.width, self.height)
        self.start_epoch = 0
        self.numFrames = self.cfg.DATASET.numFrames
        self.F = self.cfg.DATASET.numGroupFrames
        self.imgHeatmapRatio = self.cfg.DATASET.imgSize / self.cfg.DATASET.heatmapSize
        self.aspectRatio = self.imgWidth * 1.0 / self.imgHeight
        self.pixel_std = 200

    def initialize(self, LR=None): # TODO(jiahang): for now this function is not being used anymore since it's contradicted with distributed training logic.
        self.lossComputer = LossComputer(self.cfg, self.device)
        # TODO: this verification needs to be revised
        if (self.cfg.RUN.use_horovod and hvd.rank() == 0) or not self.cfg.RUN.use_horovod:
            dir_list = [self.dir, self.visdir, self.checkpoint_dir, self.result_dir, self.tensorboard_dir]
            for _dir in dir_list:
                if not os.path.isdir(_dir):
                    os.mkdir(_dir)

        if not self.cfg.RUN.eval:
            if self.cfg.TRAINING.optimizer == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
            elif self.cfg.TRAINING.optimizer == 'adam':  
                self.optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
                
            print('==========>Train set size:', len(self.trainLoader))
            print('==========>Eval set size:', len(self.evalLoader))
        print('==========>Test set size:', len(self.testLoader))
  
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=float)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspectRatio * h:
            h = w * 1.0 / self.aspectRatio
        elif w < self.aspectRatio * h:
            w = h * self.aspectRatio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=float)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def adjustLR(self, epoch):
        if epoch < self.cfg.TRAINING.warmupEpoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.cfg.TRAINING.warmupGrowth
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.cfg.TRAINING.lrDecay


    def saveStatus(self, epoch, ap, best_ap, loss_list):
        saveGroup = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ap': ap,
        }
        
        if ap >= best_ap:
            print(f'==========>Save the best model at epoch {epoch}...')
            torch.save(saveGroup, os.path.join(self.checkpoint_dir, 'model_best.pth'))
            self.saveLosslist(epoch, loss_list, 'train')
            ap = best_ap

        if epoch % 5 == 0:
            print(f'==========>Save the latest model at epoch {epoch}...')
            torch.save(saveGroup, os.path.join(self.checkpoint_dir, 'checkpoint_%d.pth'%epoch))
            self.saveLosslist(epoch, loss_list, 'train')

        return ap
        
    def _saveLosslist(self, epoch, loss_list, mode):
        with open(os.path.join(self.result_dir,'%s_loss_list_%d.json'%(mode, epoch)), 'w') as fp:
            json.dump(loss_list, fp)
    
    def loadModelWeight(self, mode, checkpoint_dir='', continue_training=False):
        assert (checkpoint_dir != '' and continue_training == True) or \
            (checkpoint_dir == '' and continue_training == False), \
            "The arguments checkpoint_dir does not align with continute_training."
        if checkpoint_dir:
            checkpoint_path = os.path.join('./logs', checkpoint_dir, 'checkpoints', '%s.pth'%mode)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, '%s.pth'%mode)
        if os.path.exists(checkpoint_path):
            print('==========>Loading the checkpoint')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if continue_training:
                print('==========>Loading the previous optimizer')
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch']
                self.start_ap = checkpoint['ap']
            print('==========>Loading the model weight from %s, saved at epoch %d' %(checkpoint_path, checkpoint['epoch']))
        else:
            print('==========>Train or evaluate the model from scratch')
        del checkpoint
        torch.cuda.empty_cache()
    
    def saveKeypoints(self, savePreds, preds, bbox, image_id, predHeatmap=None):
        
        visidx = np.ones((len(preds), self.numKeypoints, 1))
        preds = np.concatenate((preds, visidx), axis=2)
        predsigma = np.zeros((len(preds), self.numKeypoints))
        
        for j in range(len(preds)):
            block = {}
            #center, scale = self._xywh2cs(bbox[j][0], bbox[j][1], bbox[j][2] - bbox[j][0], bbox[j][3] - bbox[j][1])
            center, scale = self._xywh2cs(bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3])
            block["category_id"] = 1
            block["center"] = center.tolist()
            block["image_id"] = image_id[j].item()
            block["scale"] = scale.tolist()
            block["score"] = 1.0
            block["keypoints"] = preds[j].reshape(self.numKeypoints*3).tolist()
            if predHeatmap is not None:
                for kpts in range(self.numKeypoints):
                    predsigma[j, kpts] = predHeatmap[j, kpts].var().item() * self.heatmapSize
                block["sigma"] = predsigma[j].tolist()
            block_copy = block.copy()
            savePreds.append(block_copy)

        return savePreds

    def writeKeypoints(self, preds, phase):
        predFile = os.path.join(self.result_dir, f"{phase}_results.json")
        with open(predFile, 'w') as fp:
            json.dump(preds, fp)

    def eval(self, visualization=True, isTest=False):#should set batch size = 1
        pass
    
    def train(self):
        pass
