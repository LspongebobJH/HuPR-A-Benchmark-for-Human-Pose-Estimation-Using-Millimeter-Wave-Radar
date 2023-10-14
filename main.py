from tools import Runner
from omegaconf import OmegaConf

if __name__ == "__main__":
    cfg_cmd = OmegaConf.from_cli()
    cfg = OmegaConf.load('config/mscsa_prgcn.yaml')
    cfg = OmegaConf.merge(cfg, cfg_cmd)

    trigger = Runner(cfg)
    
    trigger.main()
        