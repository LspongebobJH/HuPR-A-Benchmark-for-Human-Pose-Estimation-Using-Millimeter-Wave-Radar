import yaml
import argparse
from tools import Runner
from collections import namedtuple
from argparse import Namespace

import wandb
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    cfg_cmd = OmegaConf.from_cli()
    cfg = OmegaConf.load('config/mscsa_prgcn.yaml')
    cfg = OmegaConf.merge(cfg, cfg_cmd)

    trigger = Runner(cfg)
    
    trigger.main()
        