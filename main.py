# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from sampler import Sampler
import ml_collections
import os


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    主函数，根据实验名称中的关键词运行指定的训练/采样任务。

    Args:
        cfg (DictConfig): Hydra配置对象，包含所有运行参数

    Raises:
        ValueError: 当实验名称中不包含任何已知关键词时
    """
    # 打印完整配置，便于调试
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # 转换配置格式
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = ml_collections.ConfigDict(cfg_dict)
    
    # 保存配置到running目录
    running_dir = os.path.join("configs", "running")
    os.makedirs(running_dir, exist_ok=True)
    # 直接使用配置文件中定义的 run_name
    config_path = os.path.join(running_dir, f"{cfg.run_name}.yaml")
    OmegaConf.save(cfg, config_path, resolve=True)
    print(f"Configuration saved to {config_path}")
    
    # 初始化训练器

    # 根据实验名称中的关键词选择运行模式
    exp_name = cfg.exp_name.lower()
    if "ae" in exp_name:
        trainer = Trainer(cfg_dict)
        trainer.train_ae()
    elif "score" in exp_name:
        trainer = Trainer(cfg_dict)
        trainer.train_score()
    elif "fsl" in exp_name:
        trainer = Trainer(cfg_dict)
        trainer.train_fsl()
    elif "sample" in exp_name:
        sampler= Sampler("ckpt",cfg_dict)
        sampler.sample(need_eval=True)
    else:
        raise ValueError(f"Unknown experiment type in name: {cfg.exp_name}. "
                        f"Name should contain one of: ae, score, fsl, sample")


if __name__ == "__main__":
    main()