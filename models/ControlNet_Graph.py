"""
Graph ControlNet实现
类似于ControlNet，创建与原始Score网络相同架构的控制分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ScoreNetwork_X import ScoreNetworkX_poincare
from models.ScoreNetwork_A import ScoreNetworkA_poincare
from utils.manifolds_utils import get_manifold


class GraphControlNet(nn.Module):
    """
    图神经网络的ControlNet实现

    架构：
    1. 原始Score网络（冻结）
    2. ControlNet分支（相同架构，可训练）
    3. 条件编码器（处理控制信号）
    4. 零卷积层（确保初始时不影响原始输出）
    """

    def __init__(self, original_score_x, original_score_adj, condition_dim=16):
        super().__init__()

        # 1. 原始Score网络（冻结）
        self.original_score_x = original_score_x
        self.original_score_adj = original_score_adj

        # 冻结原始网络参数
        for param in self.original_score_x.parameters():
            param.requires_grad = False
        for param in self.original_score_adj.parameters():
            param.requires_grad = False

        # 2. 创建ControlNet分支（相同架构）
        self.control_score_x = self._clone_network(original_score_x)
        self.control_score_adj = self._clone_network(original_score_adj)

        # 3. 基于类别原型的条件编码器
        self.condition_encoder = PrototypeConditionEncoder(
            input_dim=original_score_x.nfeat,  # 原始图特征维度
            hidden_dim=64,
            output_dim=condition_dim,
            num_classes=4,  # Letter_high数据集通常是4-way分类
        )

        # 4. 零初始化层（ControlNet的关键技巧）
        self.zero_conv_x = ZeroConv(original_score_x.nfeat)
        self.zero_conv_adj = ZeroConv(1)  # 邻接矩阵是标量

        # 5. 条件注入层
        self.condition_inject_x = nn.Linear(condition_dim, original_score_x.nfeat)
        self.condition_inject_adj = nn.Linear(condition_dim, 32)  # 调整维度

    def _clone_network(self, original_network):
        """克隆网络架构，创建新的可训练副本"""
        if isinstance(original_network, ScoreNetworkX_poincare):
            return ScoreNetworkX_poincare(
                max_feat_num=original_network.nfeat,
                depth=original_network.depth,
                nhid=original_network.nhid,
                manifold=original_network.manifold,
                edge_dim=1,  # 假设edge_dim
                GCN_type="HGCN",
            )
        elif isinstance(original_network, ScoreNetworkA_poincare):
            return ScoreNetworkA_poincare(
                max_feat_num=original_network.nfeat,  # 添加缺失参数
                max_node_num=original_network.max_node_num,  # 添加缺失参数
                nhid=original_network.nhid,  # 从原网络获取
                num_layers=original_network.num_layers,
                num_linears=original_network.num_linears,
                c_init=original_network.c_init,
                c_hid=original_network.c_hid,
                c_final=original_network.c_final,
                adim=original_network.adim,
                num_heads=getattr(original_network, "num_heads", 4),
                conv=getattr(original_network, "conv", "GCN"),
                manifold=original_network.manifold,  # 添加manifold参数
            )
        else:
            raise NotImplementedError(f"Unsupported network type: {type(original_network)}")

    def forward(self, x, adj, flags, t, graph_features=None, class_labels=None):
        """
        前向传播

        Args:
            x: 节点特征 [batch, nodes, features]
            adj: 邻接矩阵 [batch, nodes, nodes]
            flags: 节点mask
            t: 时间步
            graph_features: 图级特征 [batch, feat_dim]
            class_labels: 类别标签 [batch]
        """
        # 确保所有输入都在正确的设备上
        device = next(self.parameters()).device
        x = x.to(device)
        adj = adj.to(device)
        if flags is not None:
            flags = flags.to(device)
        if isinstance(t, torch.Tensor):
            t = t.to(device)
        else:
            t = torch.tensor(t, device=device, dtype=torch.float32)

        if graph_features is not None:
            graph_features = graph_features.to(device)
        if class_labels is not None:
            class_labels = class_labels.to(device)

        # 1. 原始Score网络输出（冻结）
        with torch.no_grad():
            original_x_score = self.original_score_x(x, adj, flags, t)
            # original_adj_score = self.original_score_adj(...)  # 需要适配参数

        # 2. 如果没有条件，直接返回原始输出
        if graph_features is None or class_labels is None:
            return original_x_score, None  # 暂时返回None for adj

        # 3. 基于类别原型的条件编码
        condition_emb = self.condition_encoder(
            graph_features, class_labels
        )  # [batch, condition_dim]

        # 4. 条件注入到特征中
        condition_x = self.condition_inject_x(condition_emb)  # [batch, feat_dim]
        condition_x = condition_x.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch, nodes, feat_dim]

        # 将条件注入到输入特征中
        x_conditioned = x + condition_x

        # 5. ControlNet分支预测
        control_x_score = self.control_score_x(x_conditioned, adj, flags, t)

        # 6. 零卷积（确保训练初期不影响原始输出）
        control_x_score = self.zero_conv_x(control_x_score)

        # 7. 最终输出 = 原始输出 + 控制输出
        final_x_score = original_x_score + control_x_score

        return final_x_score, None  # 暂时不处理adj

    def update_prototypes_from_support(self, support_x, support_labels):
        """使用支持集更新类别原型"""
        # 确保数据在正确的设备上
        device = next(self.parameters()).device
        support_x = support_x.to(device)
        support_labels = support_labels.to(device)

        # 提取图级特征（节点特征的均值）
        support_graph_features = support_x.mean(dim=1)  # [num_support, feat_dim]
        self.condition_encoder.update_prototypes(support_graph_features, support_labels)


class ConditionEncoder(nn.Module):
    """条件编码器：将控制信号编码为条件向量"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, condition):
        """
        编码条件信息

        Args:
            condition: 条件输入，可以是：
                     - 原始图的节点特征均值 [batch, feat_dim]
                     - 图的结构特征 [batch, struct_dim]
                     - 类别标签 [batch, num_classes]
        """
        return self.encoder(condition)


class ZeroConv(nn.Module):
    """零初始化卷积层（ControlNet的关键组件）"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Linear(channels, channels)
        # 零初始化权重和偏置
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class GraphControlNetTrainer:
    """ControlNet训练器"""

    def __init__(self, controlnet, config):
        self.controlnet = controlnet
        self.config = config

        # 只优化ControlNet分支的参数
        self.optimizer = torch.optim.Adam(
            [p for p in controlnet.parameters() if p.requires_grad], lr=config.get("lr", 0.0001)
        )

    def train_step(self, batch):
        """
        训练步骤

        Args:
            batch: 包含原始图和目标图的批次
                  - original_graphs: 原始图
                  - target_graphs: 目标图（增强后的图）
                  - conditions: 条件信息
        """
        original_x = batch["original_x"]
        target_x = batch["target_x"]
        adj = batch["adj"]
        condition = batch["condition"]

        # 前向传播
        predicted_score, _ = self.controlnet(
            x=original_x,
            adj=adj,
            flags=None,  # 需要生成
            t=torch.rand(original_x.size(0)),  # 随机时间步
            condition=condition,
        )

        # 计算损失（MSE）
        loss = F.mse_loss(predicted_score, target_x - original_x)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def create_graph_controlnet(original_score_x, original_score_adj):
    """工厂函数：创建图ControlNet"""
    return GraphControlNet(
        original_score_x=original_score_x, original_score_adj=original_score_adj, condition_dim=16
    )


class PrototypeConditionEncoder(nn.Module):
    """基于类别原型的条件编码器"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 原型生成网络
        self.prototype_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # 可学习的类别原型嵌入
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, output_dim) * 0.1)

        # 原型融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, graph_features, class_labels):
        """
        生成基于类别原型的条件向量

        Args:
            graph_features: 图特征 [batch, feat_dim]
            class_labels: 类别标签 [batch]
        """
        batch_size = graph_features.size(0)

        # 1. 从图特征生成实例原型
        instance_prototypes = self.prototype_generator(graph_features)  # [batch, output_dim]

        # 2. 获取对应的类别原型
        class_prototypes = self.class_prototypes[class_labels]  # [batch, output_dim]

        # 3. 融合实例原型和类别原型
        combined = torch.cat(
            [instance_prototypes, class_prototypes], dim=1
        )  # [batch, output_dim*2]
        condition_vector = self.fusion_net(combined)  # [batch, output_dim]

        return condition_vector

    def update_prototypes(self, support_features, support_labels, alpha=0.1):
        """
        使用支持集更新类别原型

        Args:
            support_features: 支持集图特征 [num_support, feat_dim]
            support_labels: 支持集标签 [num_support]
            alpha: 更新率
        """
        # 确保所有数据在相同设备上
        device = next(self.prototype_generator.parameters()).device
        support_features = support_features.to(device)
        support_labels = support_labels.to(device)

        with torch.no_grad():
            for class_id in support_labels.unique():
                # 找到该类别的所有样本
                mask = support_labels == class_id
                if mask.sum() > 0:
                    # 计算该类别的平均特征
                    class_features = support_features[mask]
                    class_mean = self.prototype_generator(class_features).mean(dim=0)

                    # 更新类别原型（移动平均）
                    self.class_prototypes[class_id] = (1 - alpha) * self.class_prototypes[
                        class_id
                    ] + alpha * class_mean
