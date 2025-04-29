# ScoreNetwork_X.py
# 该文件包含多种分数网络（Score Network）实现，支持欧式空间、双曲空间（Poincare Ball）和带注意力机制的图神经网络。

import torch
import torch.nn.functional as F
from torch import nn

from utils.manifolds_utils import exp_after_transp0
from utils.graph_utils import mask_x, pow_tensor

from layers.hyp_layers import HGCLayer, HGATLayer
from layers.layers import GCLayer, GATLayer
from layers.layers import DenseGCNConv, MLP
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
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)
        self.activation = torch.tanh

    def forward(self, x, adj, flags, t):
        # 前向传播：多层GCN+激活，拼接特征，MLP输出分数
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)
        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)
        return x


# GMH: Graph Multi-Head Attention 分数网络，支持多头注意力机制
class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()
        self.depth = depth
        self.c_init = c_init
        # 多层AttentionLayer堆叠
        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))
        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
                         use_bn=False, activate_func=F.elu)
        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        # 前向传播：多层多头注意力+激活+特征拼接+MLP输出
        adjc = pow_tensor(adj, self.c_init)
        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)
        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)
        return x
# 双曲空间（Poincare Ball等）下的分数网络，支持HGAT/HGCN等多种层

class ScoreNetworkX_poincare(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid,manifold,edge_dim,GCN_type,**kwargs):
        super(ScoreNetworkX_poincare, self).__init__()
        self.manifold = manifold
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        # 根据GCN_type选择层类型（支持欧式和双曲）
        if GCN_type == 'GCN':
            layer_type = GCLayer
        elif GCN_type == 'GAT':
            layer_type = GATLayer
        elif GCN_type == 'HGCN':
            layer_type = HGCLayer
        elif GCN_type == 'HGAT':
            layer_type = HGATLayer
        else:
            raise AttributeError
        self.layers = torch.nn.ModuleList()
        if self.manifold is not None:
            # manifold列表，支持多层双曲空间特征传递
            self.manifolds = [self.manifold]*(depth+1)
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid,self.manifolds[i],self.manifolds[i+1],edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid,self.manifolds[i],self.manifolds[i+1],edge_dim=edge_dim))
        else:
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid,edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid,edge_dim=edge_dim))
        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        # 时间嵌入和缩放，用于扩散模型的时间调制
        self.temb_net = MLP(num_layers=3, input_dim=self.nfeat, hidden_dim=2*self.nfeat, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        self.time_scale = nn.Sequential(
            nn.Linear(self.nfeat+1, self.nfeat),
            nn.ReLU(),
            nn.Linear(self.nfeat, 1)
        )

    def forward(self, x, adj, flags, t,labels, protos):
        # Check and adapt input dimension if necessary
        if x.size(-1) != self.nfeat:
            if x.size(-1) < self.nfeat:
                padding_size = self.nfeat - x.size(-1)
                padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
                print(f"Warning: Padded input x feature dim from {x.size(-1)-padding_size} to {self.nfeat}")
            else: # x.size(-1) > self.nfeat
                # Truncate or raise error - Truncating for now
                x = x[..., :self.nfeat]
                print(f"Warning: Truncated input x feature dim from {x.size(-1)} to {self.nfeat}")

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
                x_list.append(self.manifolds[i+1].logmap0(x))
            else:
                x_list.append(x)
        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        # expmap0/logmap等操作保证输出在切空间
        x = self.manifold.expmap0(x)
        x = self.manifold.logmap(xt, x)
        # 时间缩放，包含conformal factor
        x = x * self.time_scale(torch.cat([temb.repeat(1,x.size(1),1),self.manifold.lambda_x(xt,keepdim=True)],dim=-1))
        x = mask_x(x, flags)
        return x

class ScoreNetworkX_poincare_proto(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid,manifold,edge_dim,GCN_type,**kwargs):
        super().__init__()
        self.manifold = manifold
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        self.proto_weight = kwargs.get('proto_weight', 0.1)  
        # 根据GCN_type选择层类型（支持欧式和双曲）
        if GCN_type == 'GCN':
            layer_type = GCLayer
        elif GCN_type == 'GAT':
            layer_type = GATLayer
        elif GCN_type == 'HGCN':
            layer_type = HGCLayer
        elif GCN_type == 'HGAT':
            layer_type = HGATLayer
        else:
            raise AttributeError
        self.layers = torch.nn.ModuleList()
        if self.manifold is not None:
            # manifold列表，支持多层双曲空间特征传递
            self.manifolds = [self.manifold]*(depth+1)
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid,self.manifolds[i],self.manifolds[i+1],edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid,self.manifolds[i],self.manifolds[i+1],edge_dim=edge_dim))
        else:
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid,edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid,edge_dim=edge_dim))
        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        # 时间嵌入和缩放，用于扩散模型的时间调制
        self.temb_net = MLP(num_layers=3, input_dim=self.nfeat, hidden_dim=2*self.nfeat, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        self.time_scale_net = nn.Sequential(
            nn.Linear(self.nfeat+1, self.nfeat),
            nn.SiLU(),          # ⚡ 更平滑，不是 ReLU
            nn.Linear(self.nfeat, 1),
            nn.Sigmoid()        # ⚡ 限制输出在 (0,1)
        )
        self.proto_proj = MLP(          # 3 层 MLP，和 temb_net 写法一致
            num_layers=3,
            input_dim=max_feat_num,     # 10
            hidden_dim=2*max_feat_num,  # 20
            output_dim=nhid,            # 32
            use_bn=False,
            activate_func=F.elu
        )
        self.debug = kwargs.get("debug", False)
        self.global_step = 0

  # ======== 辅助：调试输出 ========
    def dbgs(self, name, tensor):
        if not self.debug:
            return
        if tensor is None:
            print(f"[DBG] {name}: None")
            return
        with torch.no_grad():
            nan_cnt = (~torch.isfinite(tensor)).sum().item()
            t_min = tensor.min().item() if torch.isfinite(tensor).any() else float('nan')
            t_max = tensor.max().item() if torch.isfinite(tensor).any() else float('nan')
            print(f"[DBG] {name}: shape={tuple(tensor.shape)}, min={t_min:.4e}, max={t_max:.4e}, nan/inf={nan_cnt}")
    # ======== forward ========
    def forward(self, x, adj, flags, t, labels, protos):
        """x: (B, N, F)  adj: (B, N, N)  t: (B,)  labels: (B, N)"""
        self.global_step += 1
        xt = x.clone()
        self.dbgs("x_init", x)

        # ---- 时间嵌入 ----
        temb = get_timestep_embedding(t, self.nfeat)  # (B, F)
        self.dbgs("temb", temb)
        x = exp_after_transp0(x, self.temb_net(temb), self.manifolds[0])
        self.dbgs("x_after_exp_transp0", x)

        # ---- 进入 list ----
        if self.manifold is not None:
            x_list = [self.manifolds[0].logmap0(x)]
        else:
            x_list = [x]
        self.dbgs("logmap0_x0", x_list[0])

        # ---- GNN layers ----
        for i in range(self.depth):
            x = self.layers[i]((x, adj))[0]
            name = f"layer{i}_out"
            self.dbgs(name, x)
            if self.manifold is not None:
                x_list.append(self.manifolds[i + 1].logmap0(x))
                self.dbgs(name + "_logmap0", x_list[-1])
            else:
                x_list.append(x)

        # ---- 拼接 & final MLP ----
        xs = torch.cat(x_list, dim=-1)
        self.dbgs("xs_cat", xs)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        self.dbgs("x_after_final", x)

        # ---- manifold delta ----
        # --- 等比缩放 ---
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        r_max = 0.5  # 把半径设得更小一些，比如0.5，比0.7更稳
        scale = torch.where(norm > r_max, r_max / (norm + 1e-6), torch.ones_like(norm))
        x = x * scale

        x = self.manifold.expmap0(x)
        self.dbgs("x_expmap0", x)
        x = self.manifold.logmap(xt, x)
        self.dbgs("x_delta", x)

        # ---- Proto 引导 ----
        if protos is not None:
            proto = protos[labels]  # (B, N, F_proto)
            proto_diff = self.manifold.logmap0(proto)
            self.dbgs("proto_diff", proto_diff)
            warm = min(1.0, self.global_step  / 500)   # 500步渐变，可以自己调
            x = x + (self.proto_weight * warm) * proto_diff
            self.dbgs("x_after_proto", x)

        # ---- 时间调制 ----
        time_input = torch.cat([
            temb.repeat_interleave(x.size(1), dim=0).view(x.size(0), x.size(1), -1),  # (B,N,F)
            self.manifold.lambda_x(xt, keepdim=True)
        ], dim=-1)
        scale_raw = self.time_scale_net(time_input)   
        scale = 0.25 + 3.75 * scale_raw 

        self.dbgs("time_scale", scale)
        x = x * scale
        self.dbgs("x_scaled", x)

        # ---- mask ----
        x = mask_x(x, flags)
        self.dbgs("x_masked", x)

        # -------- done --------
        return x

