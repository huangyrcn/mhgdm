"""
任务特定采样器 - 用于元测试任务的支持集扩充
采用ControlNet风格的设计：主模型冻结 + 轻量级适应器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from utils.loader import load_device, load_model_from_ckpt, load_ema_from_ckpt, load_sampling_fn
from utils.graph_utils import node_flags
import models.Encoders as Encoders


@dataclass
class TaskSamplerConfig:
    """任务采样器配置类 - 只包含必要参数"""

    # 设备配置
    device: str = "auto"
    device_count: int = 1

    # 数据配置
    max_node_num: int = 9
    max_feat_num: int = 5

    # 采样配置
    use_ema: bool = True
    k_augment: int = 2

    # 微调配置
    finetune_steps: int = 10
    learning_rate: float = 1e-3

    # 适应器配置
    adapter_hidden_dim: int = 64
    prototype_dim: Optional[int] = None  # 如果为None，自动推断

    # 采样器配置
    sampler_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_full_config(cls, full_config):
        """从完整配置中提取必要参数"""
        return cls(
            device=getattr(full_config, "device", "auto"),
            device_count=getattr(full_config, "device_count", 1),
            max_node_num=getattr(full_config.data, "max_node_num", 9),
            max_feat_num=getattr(full_config.data, "max_feat_num", 5),
            use_ema=getattr(full_config.sample, "use_ema", True),
            k_augment=getattr(full_config.sample, "k_augment", 2),
            finetune_steps=getattr(full_config.meta_test.task_augmentation, "finetune_steps", 10),
            learning_rate=getattr(full_config.meta_test.task_augmentation, "learning_rate", 1e-3),
            adapter_hidden_dim=getattr(
                full_config.meta_test.task_augmentation, "adapter_hidden_dim", 64
            ),
            prototype_dim=getattr(full_config.meta_test.task_augmentation, "prototype_dim", None),
            sampler_config={
                "predictor": getattr(full_config.sampler, "predictor", "reverse_diffusion"),
                "corrector": getattr(full_config.sampler, "corrector", "none"),
                "snr_x": getattr(full_config.sampler, "snr_x", 0.16),
                "snr_A": getattr(full_config.sampler, "snr_A", 0.16),
                "scale_eps_x": getattr(full_config.sampler, "scale_eps_x", 1.0),
                "scale_eps_A": getattr(full_config.sampler, "scale_eps_A", 1.0),
            },
        )


class TaskSpecificAdapter(nn.Module):
    """
    轻量级适应器模块 - ControlNet风格
    在不修改主模型的情况下，基于任务原型调整分数网络的行为
    """

    def __init__(self, score_model, prototype_dim, hidden_dim=64):
        super().__init__()
        self.score_model = score_model
        self.prototype_dim = prototype_dim

        # 原型到适应权重的映射网络
        self.prototype_to_weights = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, score_model.nf),  # 映射到分数网络的特征维度
        )

        # 适应器层 - 在分数网络的中间特征上添加调制
        self.feature_modulation = nn.Sequential(
            nn.Linear(score_model.nf, score_model.nf), nn.Tanh()  # 使用tanh确保调制在合理范围内
        )

        # 初始化为近似恒等映射
        nn.init.zeros_(self.prototype_to_weights[-1].weight)
        nn.init.zeros_(self.prototype_to_weights[-1].bias)
        nn.init.zeros_(self.feature_modulation[0].weight)
        nn.init.zeros_(self.feature_modulation[0].bias)

    def forward(self, x, t, adj=None, prototypes=None):
        """
        前向传播 - 在分数网络中注入原型信息

        Args:
            x: 输入特征
            t: 时间步
            adj: 邻接矩阵（如果是adj分数网络）
            prototypes: 类别原型 [num_classes, prototype_dim]
        """
        # 如果没有原型，直接使用原始分数网络
        if prototypes is None or prototypes.size(0) == 0:
            if adj is not None:
                return self.score_model(x, t, adj)
            else:
                return self.score_model(x, t)

        # 计算原型权重
        prototype_weights = self.prototype_to_weights(prototypes)  # [num_classes, nf]

        # 将原型权重聚合为单个调制向量
        if prototype_weights.size(0) > 1:
            modulation_vector = prototype_weights.mean(dim=0, keepdim=True)  # [1, nf]
        else:
            modulation_vector = prototype_weights  # [1, nf]

        # 扩展到batch维度
        batch_size = x.size(0)
        modulation_vector = modulation_vector.expand(batch_size, -1)  # [B, nf]

        # 使用hook机制在分数网络的中间层注入调制
        def hook_fn(module, input, output):
            # 对中间特征应用调制
            if modulation_vector.shape[-1] == output.shape[-1]:
                modulated_features = output + self.feature_modulation(
                    modulation_vector.view_as(output)
                )
                return modulated_features
            return output

        # 注册hook到分数网络的某个中间层
        hook_handle = None
        if hasattr(self.score_model, "layers") and len(self.score_model.layers) > 0:
            # 在中间层注册hook
            middle_layer_idx = len(self.score_model.layers) // 2
            hook_handle = self.score_model.layers[middle_layer_idx].register_forward_hook(hook_fn)

        try:
            # 执行前向传播
            if adj is not None:
                output = self.score_model(x, t, adj)
            else:
                output = self.score_model(x, t)

            # 如果没有注册hook，在输出上应用调制
            if hook_handle is None:
                modulation = self.feature_modulation(modulation_vector)
                if modulation.shape == output.shape:
                    output = output + modulation
                elif len(output.shape) == 3 and len(modulation.shape) == 2:
                    # 调整modulation的形状以匹配输出 [B, N, F]
                    modulation = modulation.unsqueeze(1).expand_as(output)
                    output = output + modulation

            return output

        finally:
            # 清理hook
            if hook_handle is not None:
                hook_handle.remove()


class MetaTaskSampler:
    """
    元测试任务采样器 - 专门用于支持集扩充
    采用ControlNet风格的设计，主模型冻结，使用轻量级适应器
    """

    def __init__(
        self,
        config: TaskSamplerConfig,
        score_ckpt_path: str,
        encoder_ckpt_path: Optional[str] = None,
    ):
        """
        初始化元测试任务采样器

        Args:
            config: 任务采样器配置
            score_ckpt_path: 分数网络检查点路径
            encoder_ckpt_path: 编码器检查点路径（可选）
        """
        self.config = config

        # 设置设备
        device_obj, device_str = load_device(
            device_setting=config.device,
            device_count=config.device_count,
        )
        self.device = device_obj

        # 加载预训练的分数网络（冻结）
        self._load_score_networks(score_ckpt_path)

        # 加载编码器（用于原型计算）
        if encoder_ckpt_path:
            self._load_encoder(encoder_ckpt_path)
        else:
            self.encoder = None

        # 创建适应器模块
        self._create_adapters()

        # 加载采样函数
        self._setup_sampling_function()

    def _load_score_networks(self, score_ckpt_path):
        """加载并冻结分数网络"""
        score_ckpt = torch.load(score_ckpt_path, map_location=self.device, weights_only=False)
        self.score_config = ml_collections.ConfigDict(score_ckpt["model_config"])

        # 加载分数网络
        self.mx = load_model_from_ckpt(
            self.device, score_ckpt["params_x"], score_ckpt["x_state_dict"]
        )
        self.ma = load_model_from_ckpt(
            self.device, score_ckpt["params_adj"], score_ckpt["adj_state_dict"]
        )

        # 如果使用EMA，加载EMA权重
        if self.config.use_ema and "ema_x_state_dict" in score_ckpt:
            load_ema_from_ckpt(
                self.mx, score_ckpt["ema_x_state_dict"], self.score_config.train.ema
            ).copy_to(self.mx.parameters())
            load_ema_from_ckpt(
                self.ma, score_ckpt["ema_adj_state_dict"], self.score_config.train.ema
            ).copy_to(self.ma.parameters())

        # 冻结分数网络参数
        for p in self.mx.parameters():
            p.requires_grad = False
        for p in self.ma.parameters():
            p.requires_grad = False

        self.mx.eval()
        self.ma.eval()

        # 获取流形信息
        self.manifold = getattr(self.mx, "manifold", None)

    def _load_encoder(self, encoder_ckpt_path):
        """加载编码器用于原型计算"""
        encoder_ckpt = torch.load(encoder_ckpt_path, map_location=self.device, weights_only=False)

        encoder_checkpoint_config_dict = encoder_ckpt["model_config"]
        encoder_state_dict = encoder_ckpt["encoder_state_dict"]

        # 确定编码器类名
        encoder_name = encoder_checkpoint_config_dict["model"]["encoder"]
        EncoderClass = getattr(Encoders, encoder_name)
        encoder_config_for_instantiation = ml_collections.ConfigDict(encoder_checkpoint_config_dict)

        # 创建编码器
        from utils.model_utils import extract_encoder_params

        encoder_params = extract_encoder_params(encoder_config_for_instantiation)
        self.encoder = EncoderClass(**encoder_params).to(self.device)
        self.encoder.load_state_dict(encoder_state_dict)

        # 冻结编码器参数
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def _create_adapters(self):
        """创建适应器模块"""
        # 确定原型维度
        if self.config.prototype_dim is not None:
            prototype_dim = self.config.prototype_dim
        elif self.encoder is not None:
            # 从编码器推断原型维度
            prototype_dim = getattr(self.encoder, "latent_dim", 64)
            # 如果编码器使用图级别聚合，维度可能是2倍
            if hasattr(self.encoder, "graph_level") and self.encoder.graph_level:
                prototype_dim *= 2
        else:
            # 默认原型维度，基于简单特征计算
            prototype_dim = self.config.max_feat_num + 3  # 节点特征 + 图统计

        # 创建适应器
        self.adapter_x = TaskSpecificAdapter(
            self.mx, prototype_dim, hidden_dim=self.config.adapter_hidden_dim
        ).to(self.device)

        self.adapter_adj = TaskSpecificAdapter(
            self.ma, prototype_dim, hidden_dim=self.config.adapter_hidden_dim
        ).to(self.device)

    def _setup_sampling_function(self):
        """设置采样函数"""
        # 创建临时配置对象用于采样函数
        temp_config = ml_collections.ConfigDict()
        temp_config.sampler = ml_collections.ConfigDict(self.config.sampler_config)
        temp_config.sample = ml_collections.ConfigDict({"use_ema": self.config.use_ema})

        self.sampling_fn = load_sampling_fn(
            self.score_config, temp_config.sampler, temp_config.sample, self.device, self.manifold
        )

    def augment_task(self, task, k_augment=None, finetune_steps=None, learning_rate=None):
        """
        增强元测试任务的支持集

        Args:
            task: 元测试任务字典
            k_augment: 每个支持样本生成的增强样本数量（可选，使用配置默认值）
            finetune_steps: 适应器微调步数（可选，使用配置默认值）
            learning_rate: 学习率（可选，使用配置默认值）

        Returns:
            augmented_task: 增强后的任务字典
        """
        # 使用传入参数或配置默认值
        k_augment = k_augment if k_augment is not None else self.config.k_augment
        finetune_steps = (
            finetune_steps if finetune_steps is not None else self.config.finetune_steps
        )
        learning_rate = learning_rate if learning_rate is not None else self.config.learning_rate

        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_labels = task["support_set"]["label"].to(self.device)

        # 第一步：计算类别原型
        prototypes = self._compute_class_prototypes(support_x, support_adj, support_labels)

        # 第二步：微调适应器
        if finetune_steps > 0:
            self._finetune_adapters(
                support_x, support_adj, support_labels, prototypes, finetune_steps, learning_rate
            )

        # 第三步：基于原型生成增强样本
        augmented_samples = self._generate_augmented_samples(
            support_x, support_adj, support_labels, prototypes, k_augment
        )

        # 第四步：构造增强后的任务
        augmented_task = self._build_augmented_task(task, augmented_samples)

        return augmented_task

    def _compute_class_prototypes(self, support_x, support_adj, support_labels):
        """计算类别原型"""
        if self.encoder is None:
            return self._compute_simple_prototypes(support_x, support_adj, support_labels)

        # 使用编码器计算嵌入
        node_masks = torch.stack([node_flags(adj) for adj in support_adj])

        with torch.no_grad():
            posterior = self.encoder(support_x, support_adj, node_masks)
            embeddings = posterior.mode()

            # 确保是图级别嵌入
            if embeddings.dim() == 3:
                if embeddings.size(1) == 1:
                    embeddings = embeddings.squeeze(1)
                else:
                    # 图级别聚合
                    valid_nodes = node_masks.float().unsqueeze(-1)
                    masked_embeddings = embeddings * valid_nodes

                    mean_emb = masked_embeddings.sum(dim=1) / valid_nodes.sum(dim=1)

                    masked_embeddings_for_max = masked_embeddings.clone()
                    invalid_mask = (valid_nodes == 0).expand_as(masked_embeddings)
                    masked_embeddings_for_max[invalid_mask] = float("-inf")
                    max_emb = masked_embeddings_for_max.max(dim=1).values

                    embeddings = torch.cat([mean_emb, max_emb], dim=-1)

            # 如果使用流形，映射到切空间
            if self.encoder.manifold is not None:
                embeddings = self.encoder.manifold.logmap0(embeddings)

        # 按标签计算原型
        unique_labels = torch.unique(support_labels).sort()[0]
        prototypes = []

        for label in unique_labels:
            mask = support_labels == label
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                class_prototype = class_embeddings.mean(dim=0, keepdim=True)
                prototypes.append(class_prototype)

        if prototypes:
            prototypes = torch.cat(prototypes, dim=0)
        else:
            # 默认零原型
            emb_dim = embeddings.shape[-1] if embeddings.numel() > 0 else 64
            prototypes = torch.zeros(1, emb_dim, device=support_x.device)

        return prototypes

    def _compute_simple_prototypes(self, support_x, support_adj, support_labels):
        """计算简单的图级别原型（不使用编码器）"""
        batch_size = support_x.size(0)

        # 为每个图计算简单特征
        graph_features = []
        for i in range(batch_size):
            x_i = support_x[i]  # [max_nodes, feat_dim]
            adj_i = support_adj[i]  # [max_nodes, max_nodes]

            # 有效节点掩码
            node_mask = node_flags(adj_i)
            valid_nodes = node_mask.sum().item()

            if valid_nodes > 0:
                # 节点特征均值
                mean_node_feat = x_i[node_mask].mean(dim=0)

                # 图统计特征
                degree_stats = adj_i[node_mask][:, node_mask].sum(dim=1).float()
                mean_degree = degree_stats.mean()
                max_degree = degree_stats.max()

                # 组合特征
                graph_feat = torch.cat(
                    [
                        mean_node_feat,
                        torch.tensor(
                            [valid_nodes, mean_degree, max_degree], device=support_x.device
                        ),
                    ]
                )
            else:
                # 零图
                feat_dim = support_x.size(-1)
                graph_feat = torch.zeros(feat_dim + 3, device=support_x.device)

            graph_features.append(graph_feat)

        graph_features = torch.stack(graph_features)  # [batch_size, feat_dim]

        # 按标签计算原型
        unique_labels = torch.unique(support_labels).sort()[0]
        prototypes = []

        for label in unique_labels:
            mask = support_labels == label
            if mask.sum() > 0:
                class_features = graph_features[mask]
                class_prototype = class_features.mean(dim=0, keepdim=True)
                prototypes.append(class_prototype)

        if prototypes:
            prototypes = torch.cat(prototypes, dim=0)
        else:
            prototypes = torch.zeros(1, graph_features.size(-1), device=support_x.device)

        return prototypes

    def _finetune_adapters(
        self, support_x, support_adj, support_labels, prototypes, steps, learning_rate
    ):
        """微调适应器以适应当前任务"""
        # 设置适应器为训练模式
        self.adapter_x.train()
        self.adapter_adj.train()

        # 优化器
        optimizer = torch.optim.Adam(
            list(self.adapter_x.parameters()) + list(self.adapter_adj.parameters()),
            lr=learning_rate,
        )

        for step in range(steps):
            optimizer.zero_grad()

            # 随机时间步
            t = torch.randint(
                0, self.score_config.model.num_timesteps, (support_x.size(0),), device=self.device
            )

            # 添加噪声
            noise_x = torch.randn_like(support_x)
            noise_adj = torch.randn_like(support_adj)

            # 获取噪声调度
            alpha_t = self._get_alpha_t(t)
            alpha_t = alpha_t.view(-1, 1, 1)  # [B, 1, 1]

            # 噪声化
            x_noisy = torch.sqrt(alpha_t) * support_x + torch.sqrt(1 - alpha_t) * noise_x

            alpha_t_adj = alpha_t.view(-1, 1, 1)  # [B, 1, 1] for adj
            adj_noisy = (
                torch.sqrt(alpha_t_adj) * support_adj + torch.sqrt(1 - alpha_t_adj) * noise_adj
            )

            # 使用适应器预测噪声
            predicted_noise_x = self.adapter_x(x_noisy, t, prototypes=prototypes)
            predicted_noise_adj = self.adapter_adj(adj_noisy, t, adj=x_noisy, prototypes=prototypes)

            # 重构损失
            loss_x = F.mse_loss(predicted_noise_x, noise_x)
            loss_adj = F.mse_loss(predicted_noise_adj, noise_adj)
            loss = loss_x + loss_adj

            loss.backward()
            optimizer.step()

        # 恢复评估模式
        self.adapter_x.eval()
        self.adapter_adj.eval()

    def _get_alpha_t(self, t):
        """获取噪声调度参数"""
        # 简化的噪声调度，实际应该与训练时一致
        num_timesteps = self.score_config.model.num_timesteps
        beta_1 = 1e-4
        beta_T = 2e-2
        betas = torch.linspace(beta_1, beta_T, num_timesteps, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod[t]

    def _generate_augmented_samples(
        self, support_x, support_adj, support_labels, prototypes, k_augment
    ):
        """基于原型生成增强样本"""
        batch_size = support_x.size(0)
        generated_samples = {"x": [], "adj": [], "labels": []}

        with torch.no_grad():
            for _ in range(k_augment):
                # 为每个支持样本生成一个增强样本
                shape_x = (batch_size, self.config.max_node_num, self.config.max_feat_num)
                shape_adj = (batch_size, self.config.max_node_num, self.config.max_node_num)

                # 创建增强的采样函数，使用适应器
                def adapted_sampling_fn(mx, ma, shape_x, shape_adj, labels, task_prototypes):
                    # 使用适应器替换原始模型
                    return self.sampling_fn(
                        self.adapter_x,
                        self.adapter_adj,
                        shape_x,
                        shape_adj,
                        labels,
                        task_prototypes,
                    )

                # 生成样本
                x_gen, adj_gen = adapted_sampling_fn(
                    self.mx, self.ma, shape_x, shape_adj, support_labels, prototypes
                )

                generated_samples["x"].append(x_gen)
                generated_samples["adj"].append(adj_gen)
                generated_samples["labels"].append(support_labels)

        return generated_samples

    def _build_augmented_task(self, original_task, augmented_samples):
        """构建增强后的任务"""
        # 原始支持集
        original_support_x = original_task["support_set"]["x"]
        original_support_adj = original_task["support_set"]["adj"]
        original_support_labels = original_task["support_set"]["label"]

        # 合并原始样本和增强样本
        all_support_x = [original_support_x.to(self.device)]
        all_support_adj = [original_support_adj.to(self.device)]
        all_support_labels = [original_support_labels.to(self.device)]

        for x_gen, adj_gen, labels_gen in zip(
            augmented_samples["x"], augmented_samples["adj"], augmented_samples["labels"]
        ):
            all_support_x.append(x_gen)
            all_support_adj.append(adj_gen)
            all_support_labels.append(labels_gen)

        # 合并张量
        augmented_support_x = torch.cat(all_support_x, dim=0)
        augmented_support_adj = torch.cat(all_support_adj, dim=0)
        augmented_support_labels = torch.cat(all_support_labels, dim=0)

        # 构建增强后的任务
        augmented_task = {
            "support_set": {
                "x": augmented_support_x.cpu(),
                "adj": augmented_support_adj.cpu(),
                "label": augmented_support_labels.cpu(),
            },
            "query_set": original_task["query_set"],  # 查询集保持不变
            "N_way": original_task.get("N_way", None),
            "K_shot": original_task.get("K_shot", None),
            "R_query": original_task.get("R_query", None),
            "append_count": original_task.get("append_count", 0),
        }

        return augmented_task


# 工厂函数
def create_meta_task_sampler(
    config: TaskSamplerConfig, score_ckpt_path: str, encoder_ckpt_path: Optional[str] = None
):
    """
    工厂函数：创建元测试任务采样器

    Args:
        config: TaskSamplerConfig 配置对象
        score_ckpt_path: 分数网络检查点路径
        encoder_ckpt_path: 编码器检查点路径（可选）

    Returns:
        MetaTaskSampler 实例
    """
    return MetaTaskSampler(config, score_ckpt_path, encoder_ckpt_path)
