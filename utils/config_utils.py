import yaml
import os
import re
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Any, Union


def resolve_template_variables(value: str, context: Dict[str, Any] = None) -> str:
    """解析配置中的模板变量"""
    if not isinstance(value, str):
        return value

    # 解析 ${now} 为当前时间戳
    if "${now}" in value:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        value = value.replace("${now}", timestamp)

    # 如果提供了上下文，用上下文变量替换模板
    if context:
        for key, val in context.items():
            placeholder = f"${{{key}}}"
            if placeholder in value:
                value = value.replace(placeholder, str(val))

    return value


def process_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """递归处理配置字典，解析所有模板变量"""
    processed = {}

    # 第一步：处理基本模板变量
    for key, value in config_dict.items():
        if isinstance(value, dict):
            processed[key] = process_config_dict(value)
        elif isinstance(value, str):
            processed[key] = resolve_template_variables(value)
        else:
            processed[key] = value

    # 第二步：生成运行时上下文
    context = {}

    # 创建 run_name 上下文变量
    if "exp_name" in processed and "timestamp" in processed:
        context["run_name"] = f"{processed['exp_name']}_{processed['timestamp']}"
    elif "exp_name" in processed:
        # 如果没有timestamp，使用当前时间
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed["timestamp"] = timestamp
        context["run_name"] = f"{processed['exp_name']}_{timestamp}"

    # 添加已解析的变量到上下文
    context.update(processed)

    # 第三步：用上下文解析复合模板
    def resolve_with_context(obj):
        if isinstance(obj, dict):
            return {k: resolve_with_context(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            return resolve_template_variables(obj, context)
        else:
            return obj

    return resolve_with_context(processed)


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

    # 处理模板变量
    config_dict = process_config_dict(config_dict)

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
