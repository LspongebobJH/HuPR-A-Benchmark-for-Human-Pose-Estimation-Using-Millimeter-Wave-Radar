import sys
sys.path.append('/home/ubuntu/hupr')

from datasets.dataset import getDataset, collate_fn
from omegaconf import OmegaConf

import numpy as np
import multiprocessing as mp

# from tqdm.contrib import tzip

# def func(path1, path2):
#     try:
#         np.load(path1)
#     except Exception as e:
#         error_list.append(path1)
    
#     try:
#         np.load(path2)
#     except Exception as e:
#         error_list.append(path2)
#     return error_list
def func(c):
    a, b = c[0], c[1]
    return a*b

cfg_cmd = OmegaConf.from_cli()
cfg = OmegaConf.load('config/mscsa_prgcn.yaml')
cfg = OmegaConf.merge(cfg, cfg_cmd)

error_list = []
dataset = getDataset('train', cfg)

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]

pool = mp.Pool(4)
# pool.map(func, zip(dataset.VRDAEPaths_hori, dataset.VRDAEPaths_vert))
res = pool.map(func, zip(a, b))
pool.join()
pass
    

# dataset = getDataset('val', cfg)
# for path1, path2 in zip(dataset.VRDAEPaths_hori, dataset.VRDAEPaths_vert):
#     try:
#         np.load(path1)
#     except Exception as e:
#         error_list.append(path1)
    
#     try:
#         np.load(path2)
#     except Exception as e:
#         error_list.append(path2)

# dataset = getDataset('test', cfg)
# for path1, path2 in zip(dataset.VRDAEPaths_hori, dataset.VRDAEPaths_vert):
#     try:
#         np.load(path1)
#     except Exception as e:
#         error_list.append(path1)
    
#     try:
#         np.load(path2)
#     except Exception as e:
#         error_list.append(path2)

pass
