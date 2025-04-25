import hydra
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer
import ml_collections


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Convert DictConfig to a standard Python dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = ml_collections.ConfigDict(cfg_dict)
    trainer = Trainer(cfg_dict)  # Pass the dictionary
    trainer.trai    n_score()


if __name__ == "__main__":
    main()
