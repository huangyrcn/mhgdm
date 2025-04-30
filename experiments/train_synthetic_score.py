import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import yaml
from trainer.trainer import Trainer
import yaml
import ml_collections




if __name__ == "__main__":
    yaml_config_path = pathlib.Path(__file__).parent.resolve().parent / "configs" / "enzymes_configs" / "enzymes_train_score.yaml"

    with open(yaml_config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = ml_collections.ConfigDict(config_dict)
    trainer = Trainer(config)
    trainer.train_score()
