import sys
sys.path.append('/home/ubuntu/hupr')

from datasets.dataset import getDataset, collate_fn
from omegaconf import OmegaConf

import numpy as np
import multiprocessing as mp
import pickle

def func(paths):
    path1, path2 = paths[0], paths[1]
    error_list = []
    try:
        np.load(path1)
        print(f"Finish {path1}")
    except Exception as e:
        error_list.append(path1)
        print(f"Error {path1}, {e}")
    try:
        np.load(path2)
        print(f"Finish {path2}")
    except Exception as e:
        error_list.append(path2)
        print(f"Error {path2}, {e}")
    return error_list

cfg_cmd = OmegaConf.from_cli()
cfg = OmegaConf.load('config/mscsa_prgcn.yaml')
cfg = OmegaConf.merge(cfg, cfg_cmd)

error_list = []
pool = mp.Pool(8)

print('=== test ===')
dataset = getDataset('test', cfg)
res = pool.map(func, zip(dataset.VRDAEPaths_hori, dataset.VRDAEPaths_vert))
error_list.extend(res)

print('=== train ===')
dataset = getDataset('train', cfg)
res = pool.map(func, zip(dataset.VRDAEPaths_hori, dataset.VRDAEPaths_vert))
error_list.extend(res)

print('=== val ===')
dataset = getDataset('val', cfg)
res = pool.map(func, zip(dataset.VRDAEPaths_hori, dataset.VRDAEPaths_vert))
error_list.extend(res)

pool.close()
pool.join()
    
with open('error_list.pkl', 'wb') as f:
    pickle.dump(error_list, f)
