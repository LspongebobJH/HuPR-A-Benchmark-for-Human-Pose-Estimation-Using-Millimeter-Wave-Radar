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
from ray.tune import TuneConfig
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import setup_wandb

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    cfg_cmd = OmegaConf.from_cli()
    cfg = OmegaConf.load('config/mscsa_prgcn.yaml')
    cfg = OmegaConf.merge(cfg, cfg_cmd)

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
            # scaling_config = ScalingConfig(
            #     trainer_resources={'CPU': 80, 'GPU': 8}, use_gpu=True, 
            #     num_workers=cfg.RUN.num_ray_workers, resources_per_worker={'CPU': 8, 'GPU': 1}
            # )
            scaling_config = ScalingConfig(
                trainer_resources={'CPU': 10, 'GPU': 1}, use_gpu=True, 
                num_workers=cfg.RUN.num_ray_workers, resources_per_worker={'CPU': 8, 'GPU': 1}
            )
            # run_config = RunConfig(
            #     name=cfg.RUN.project, local_dir='./logs', log_to_file="output.log", verbose=0,
            #     checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute='ap', checkpoint_score_order='max')
            # )
            run_config = RunConfig(
                name=cfg.RUN.project, local_dir='./logs', log_to_file="output.log",
                checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute='ap', checkpoint_score_order='max')
            )
            # tune_config = TuneConfig(resources_per_trial=)
            trainer = TorchTrainer(
                train_loop_per_worker=trigger.main,
                scaling_config=scaling_config,
                run_config=run_config,
                # tune_config=tune_config
            )
            results = trainer.fit()
            print(f"Last results {results.metrics}")
    