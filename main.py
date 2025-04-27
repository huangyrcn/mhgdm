# main.py
"""
入口脚本，用于运行不同的训练/采样任务。

这个脚本是项目的统一入口点，通过hydra配置系统来管理不同的实验。
根据实验名称中的关键词来决定运行哪个函数：
- ae: 训练自编码器模型
- score: 训练分数模型
- fsl: 训练少样本学习模型
- sample: 使用训练好的模型进行采样

使用方法：
1. 通过配置文件指定实验名称：
   exp_name: "proto_guide_ae_model"  # 在config.yaml中设置

2. 或通过命令行参数指定：
   python main.py exp_name=proto_guide_score_model

配置系统：
- 使用hydra管理配置
- 配置文件位于configs目录
- 支持配置继承和覆盖
- 每次运行的配置会保存到configs/running/{run_name}.yaml
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
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
    trainer = Trainer(cfg_dict)
    
    # 根据实验名称中的关键词选择运行模式
    exp_name = cfg.exp_name.lower()
    if "ae" in exp_name:
        trainer.train_ae()
    elif "score" in exp_name:
        trainer.train_score()
    elif "fsl" in exp_name:
        trainer.train_fsl()
    elif "sample" in exp_name:
        trainer.sample()
    else:
        raise ValueError(f"Unknown experiment type in name: {cfg.exp_name}. "
                        f"Name should contain one of: ae, score, fsl, sample")


if __name__ == "__main__":
    main()
