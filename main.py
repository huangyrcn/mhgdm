# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
# Ensure the import points to the correct location after refactoring
from trainer import Trainer 
# Assuming Sampler is still in the root directory or adjust import if moved
from sampler import Sampler 
import ml_collections
import os


@hydra.main(config_path="configs", config_name="ae", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = ml_collections.ConfigDict(cfg_dict)
    
    running_dir = os.path.join("configs", "running")
    os.makedirs(running_dir, exist_ok=True)

    config_path = os.path.join(running_dir, f"{cfg.run_name}.yaml")
    OmegaConf.save(cfg, config_path, resolve=True)
    print(f"Configuration saved to {config_path}")
    
    exp_name = cfg.exp_name.lower()
    if "ae" in exp_name:
        trainer = Trainer(cfg_dict) # Trainer initialization remains the same
        trainer.train_ae()
    elif "score" in exp_name:
        trainer = Trainer(cfg_dict) # Trainer initialization remains the same
        trainer.train_score()
    elif "fsl" in exp_name:
        trainer = Trainer(cfg_dict) # Trainer initialization remains the same
        trainer.train_fsl()
    elif "sample" in exp_name:
      
        sampler= Sampler(cfg_dict) 
        sampler.sample() 
    else:
        raise ValueError(f"Unknown experiment type in name: {cfg.exp_name}. "
                        f"Name should contain one of: ae, score, fsl, sample")


if __name__ == "__main__":
    main()