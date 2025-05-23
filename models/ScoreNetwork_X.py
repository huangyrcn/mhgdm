# ScoreNetwork_X.py
# 该文件包含多种分数网络（Score Network）实现，支持欧式空间、双曲空间（Poincare Ball）和带注意力机制的图神经网络。

import torch
import torch.nn.functional as F
from torch import nn

from utils.manifolds_utils import exp_after_transp0
from utils.graph_utils import mask_x, pow_tensor

from layers.hyp_layers import HGCLayer, HGATLayer
from layers.euc_layers import GCLayer, GATLayer
from models.layers import DenseGCNConv, MLP
from utils.model_utils import get_timestep_embedding
from layers.attention import AttentionLayer


# 欧式空间下的分数网络，基于多层GCN实现
class ScoreNetworkX(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid):
        super(ScoreNetworkX, self).__init__()
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        # 堆叠多层GCN
        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))
        # 拼接所有层输出后，最终MLP输出分数
        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )
        self.activation = torch.tanh

    def forward(self, x, adj, flags, t):
        # 前向传播：多层GCN+激活，拼接特征，MLP输出分数
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)
        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)
        return x


# GMH: Graph Multi-Head Attention 分数网络，支持多头注意力机制
class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(
        self,
        max_feat_num,
        depth,
        nhid,
        num_linears,
        c_init,
        c_hid,
        c_final,
        adim,
        num_heads=4,
        conv="GCN",
    ):
        super().__init__()
        self.depth = depth
        self.c_init = c_init
        # 多层AttentionLayer堆叠
        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(
                    AttentionLayer(
                        num_linears, max_feat_num, nhid, nhid, c_init, c_hid, num_heads, conv
                    )
                )
            elif _ == self.depth - 1:
                self.layers.append(
                    AttentionLayer(num_linears, nhid, adim, nhid, c_hid, c_final, num_heads, conv)
                )
            else:
                self.layers.append(
                    AttentionLayer(num_linears, nhid, adim, nhid, c_hid, c_hid, num_heads, conv)
                )
        fdim = max_feat_num + depth * nhid
        self.final = MLP(
            num_layers=3,
            input_dim=fdim,
            hidden_dim=2 * fdim,
            output_dim=max_feat_num,
            use_bn=False,
            activate_func=F.elu,
        )
        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        # 前向传播：多层多头注意力+激活+特征拼接+MLP输出
        adjc = pow_tensor(adj, self.c_init)
        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)
        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)
        return x


# 双曲空间（Poincare Ball等）下的分数网络，支持HGAT/HGCN等多种层


class ScoreNetworkX_poincare(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, manifold, edge_dim, GCN_type, **kwargs):
        super(ScoreNetworkX_poincare, self).__init__()
        self.manifold = manifold
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        # 根据GCN_type选择层类型（支持欧式和双曲）
        if GCN_type == "GCN":
            layer_type = GCLayer
        elif GCN_type == "GAT":
            layer_type = GATLayer
        elif GCN_type == "HGCN":
            layer_type = HGCLayer
        elif GCN_type == "HGAT":
            layer_type = HGATLayer
        else:
            raise AttributeError
        self.layers = torch.nn.ModuleList()
        if self.manifold is not None:
            # manifold列表，支持多层双曲空间特征传递
            self.manifolds = [self.manifold] * (depth + 1)
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(
                        layer_type(
                            self.nfeat,
                            self.nhid,
                            self.manifolds[i],
                            self.manifolds[i + 1],
                            edge_dim=edge_dim,
                        )
                    )
                else:
                    self.layers.append(
                        layer_type(
                            self.nhid,
                            self.nhid,
                            self.manifolds[i],
                            self.manifolds[i + 1],
                            edge_dim=edge_dim,
                        )
                    )
        else:
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid, edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid, edge_dim=edge_dim))
        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )
        # 时间嵌入和缩放，用于扩散模型的时间调制
        self.temb_net = MLP(
            num_layers=3,
            input_dim=self.nfeat,
            hidden_dim=2 * self.nfeat,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )
        self.time_scale = nn.Sequential(
            nn.Linear(self.nfeat + 1, self.nfeat), nn.ReLU(), nn.Linear(self.nfeat, 1)
        )

    def forward(self, x, adj, flags, t, labels, protos):
        # Check and adapt input dimension if necessary
        if x.size(-1) != self.nfeat:
            if x.size(-1) < self.nfeat:
                padding_size = self.nfeat - x.size(-1)
                padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
            else:  # x.size(-1) > self.nfeat
                # Truncate or raise error - Truncating for now
                x = x[..., : self.nfeat]

        # 前向传播：双曲空间特征变换+多层HGAT/HGCN+时间调制
        xt = x.clone()
        temb = get_timestep_embedding(t, self.nfeat)
        # Note: temb_net input dim is self.nfeat, output dim is self.nfeat
        temb_output = self.temb_net(temb)

        # Ensure temb_output is correctly shaped for broadcasting if needed by exp_after_transp0
        # Assuming exp_after_transp0 handles broadcasting or expects [B, N, D]
        # If temb_output is [B, D], it might need repeating: temb_output.unsqueeze(1).repeat(1, x.size(1), 1)
        # However, let's assume geoopt handles the [B, D] case for v_at_0 in transp0

        x = exp_after_transp0(x, temb_output, self.manifolds[0])
        if self.manifold is not None:
            x_list = [self.manifolds[0].logmap0(x)]
        else:
            x_list = [x]
        for i in range(self.depth):
            x = self.layers[i]((x, adj))[0]
            if self.manifold is not None:
                x_list.append(self.manifolds[i + 1].logmap0(x))
            else:
                x_list.append(x)
        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        # expmap0/logmap等操作保证输出在切空间
        x = self.manifold.expmap0(x)
        x = self.manifold.logmap(xt, x)
        # 时间缩放，包含conformal factor
        x = x * self.time_scale(
            torch.cat(
                [temb.repeat(1, x.size(1), 1), self.manifold.lambda_x(xt, keepdim=True)], dim=-1
            )
        )
        x = mask_x(x, flags)
        return x


class ScoreNetworkX_poincare_proto(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, manifold, edge_dim, GCN_type, **kwargs):
        super(ScoreNetworkX_poincare_proto, self).__init__()
        self.manifold = manifold
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        self.proto_weight = kwargs.get("proto_weight", 0.1)  # Default weight for proto guidance
        if GCN_type == "GCN":
            layer_type = GCLayer
        elif GCN_type == "GAT":
            layer_type = GATLayer
        elif GCN_type == "HGCN":
            layer_type = HGCLayer
        elif GCN_type == "HGAT":
            layer_type = HGATLayer
        else:
            raise AttributeError
        self.layers = torch.nn.ModuleList()

        # 构建图卷积层
        if self.manifold is not None:
            self.manifolds = [self.manifold] * (depth + 1)
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(
                        layer_type(
                            self.nfeat,
                            self.nhid,
                            self.manifolds[i],
                            self.manifolds[i + 1],
                            edge_dim=edge_dim,
                        )
                    )
                else:
                    self.layers.append(
                        layer_type(
                            self.nhid,
                            self.nhid,
                            self.manifolds[i],
                            self.manifolds[i + 1],
                            edge_dim=edge_dim,
                        )
                    )
        else:
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid, edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid, edge_dim=edge_dim))

        # MLP层，用于最终输出
        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )

        # 时间嵌入网络
        self.temb_net = MLP(
            num_layers=3,
            input_dim=self.nfeat,
            hidden_dim=2 * self.nfeat,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )
        self.time_scale = nn.Sequential(
            nn.Linear(self.nfeat + 1, self.nfeat), nn.ReLU(), nn.Linear(self.nfeat, 1)
        )

    def forward(self, x, adj, flags, t, labels, protos):
        # 如果输入特征的维度小于预设维度，则进行补全
        if x.size(-1) != self.nfeat:
            if x.size(-1) < self.nfeat:
                padding_size = self.nfeat - x.size(-1)
                padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., : self.nfeat]

        xt = x.clone()
        temb = get_timestep_embedding(t, self.nfeat)
        temb_output = self.temb_net(temb)

        # 进行时间调制处理
        x = exp_after_transp0(x, temb_output, self.manifolds[0])

        if self.manifold is not None:
            x_list = [self.manifolds[0].logmap0(x)]
        else:
            x_list = [x]

        # 计算原型相似性并融合
        if protos is not None:
            proto = protos[labels]  # (B, N, 10)
            if self.manifold is not None:
                # Logmap 将 proto 和 x 映射到切空间
                log_proto_x = self.manifolds[0].logmap(
                    proto, x
                )  # (B, N, D) - The tangent vector at x in the manifold

                # 进行原型的移动操作
                x = self.manifold.expmap(
                    x, log_proto_x * self.proto_weight
                )  # Move along the geodesic towards proto
            else:
                # If the manifold is None, fall back to Euclidean space for the movement (as in original code)
                x = x + self.proto_weight * (proto - x)

        # 通过图卷积层传播特征
        for i in range(self.depth):
            x = self.layers[i]((x, adj))[0]
            if self.manifold is not None:
                x_list.append(self.manifolds[i + 1].logmap0(x))
            else:
                x_list.append(x)

        # 拼接所有层的输出
        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        # 双曲空间的映射操作
        x = self.manifold.expmap0(x)
        x = self.manifold.logmap(xt, x)

        # 时间缩放
        x = x * self.time_scale(
            torch.cat(
                [temb.repeat(1, x.size(1), 1), self.manifold.lambda_x(xt, keepdim=True)], dim=-1
            )
        )

        # 掩码应用，忽略不需要的节点
        x = mask_x(x, flags)

        return x
