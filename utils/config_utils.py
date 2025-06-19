import yaml
import os
from types import SimpleNamespace
from typing import Dict, Any, Union


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """递归地将字典转换为SimpleNamespace对象"""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def namespace_to_dict(ns: SimpleNamespace) -> Dict[str, Any]:
    """递归地将SimpleNamespace对象转换为字典"""
    result = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            result[k] = namespace_to_dict(v)
        else:
            result[k] = v
    return result


class Config(SimpleNamespace):
    """配置类，继承自SimpleNamespace，添加便利方法"""

    def to_dict(self):
        """转换为字典"""
        return namespace_to_dict(self)


def load_config(config_path: str) -> Config:
    """
    从YAML文件加载配置并返回Config对象

    Args:
        config_path: 配置文件路径

    Returns:
        Config: 配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 转换为Config对象
    return dict_to_config(config_dict)


def dict_to_config(d: Dict[str, Any]) -> Config:
    """递归地将字典转换为Config对象"""
    config = Config()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(config, k, dict_to_config(v))
        else:
            setattr(config, k, v)
    return config


def save_config(config: Union[Config, SimpleNamespace, Dict[str, Any]], save_path: str):
    """
    保存配置到YAML文件

    Args:
        config: 配置对象 (Config/SimpleNamespace) 或字典
        save_path: 保存路径
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为字典
    if isinstance(config, (Config, SimpleNamespace)):
        config_dict = namespace_to_dict(config)
    else:
        config_dict = config

    # 保存到YAML文件
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
