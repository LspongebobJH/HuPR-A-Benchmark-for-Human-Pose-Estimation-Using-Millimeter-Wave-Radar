import sys
sys.path.append('/home/ubuntu/hupr')

import os
import cv2
import math
import torch as th
import numpy as np
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.metrics import get_max_preds
from misc.utils import generateTarget
from datasets.dataset import HuPR3D_horivert

import yaml

class obj(object):
    # codes adopted from main.py
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

if __name__ == '__main__':
    with open('./config/mscsa_prgcn.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    
    dataset = HuPR3D_horivert('test', cfg)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # slice 1, 15 [20, 90)
    # slice 2, 15 [120, 190)
    # slice 3, 15 [230, 280)

    tL = [[20, 90], [120, 190], [230, 280]]
    for i, t in enumerate(tL):
        slice = [dataset[i] for i in range(*t)]
        hori = [slice_['VRDAEmap_hori'].flatten() for slice_ in slice]
        vert = [slice_['VRDAEmap_vert'].flatten() for slice_ in slice]
        hori, vert = th.stack(hori), th.stack(vert)
        hori_left = th.concat([th.zeros(1, hori.shape[1]), hori], dim=0)
        hori_right = th.concat([hori, th.zeros(1, hori.shape[1])], dim=0)
        vert_left = th.concat([th.zeros(1, vert.shape[1]), vert], dim=0)
        vert_right = th.concat([vert, th.zeros(1, vert.shape[1])], dim=0)
        print(f"input successive frame relative difference 15 slice {i}")
        print(f"hori: {(hori_left - hori_right)[1:-1].abs().mean() / hori.abs().mean() * 100:.2f}%")
        print(f"vert: {(vert_left - vert_right)[1:-1].abs().mean() / vert.abs().mean() * 100:.2f}%")
        print()
        
