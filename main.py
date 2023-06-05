import yaml
import argparse
from tools import Runner
from collections import namedtuple
from argparse import Namespace

import ray
from ray.air import session, Checkpoint, CheckpointConfig
from ray import train
import ray.train.torch as rt
from ray.air.config import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import setup_wandb

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config/", config_name="mscsa_prgcn.yaml")
def main(cfg):
    trigger = Runner(cfg)
    vis = False if cfg.RUN.visdir == 'none' else True
    if cfg.RUN.test:
        trigger.loadModelWeight('model_best')
        trigger.test(visualization=vis)
    else:
        if cfg.RUN.train_load_checkpoint:
            trigger.loadModelWeight('checkpoint')
        if not cfg.RUN.use_ray:
            trigger.main()
        else:
            ray.init()
            scaling_config = ScalingConfig(
                trainer_resources={'CPU': 16, 'GPU': 1}, use_gpu=True, 
                num_workers=cfg.RUN.num_ray_workers, resources_per_worker={'CPU': 4, 'GPU': 1}
            )
            run_config = RunConfig(
                name=cfg.RUN.project, local_dir='./logs', log_to_file="output.log", verbose=0,
                checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute='ap', checkpoint_score_order='max')
            )
            trainer = TorchTrainer(
                train_loop_per_worker=trigger.main,
                scaling_config=scaling_config,
                run_config=run_config
            )
            results = trainer.fit()
            print(f"Last results {results.metrics}")
if __name__ == "__main__":
    main()
    