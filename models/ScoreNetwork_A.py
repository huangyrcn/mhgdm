import torch
from torch import nn
import torch.nn.functional as F
from layers.CentroidDistance import CentroidDistance
from layers.layers import DenseGCNConv, MLP
from utils.model_utils import get_timestep_embedding
from utils.graph_utils import mask_adjs, pow_tensor
from layers.attention import AttentionLayer, HAttentionLayer
from utils.graph_utils import mask_x, node_feature_to_matrix

class BaselineNetworkLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

        super(BaselineNetworkLayer, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2*conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
                            use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)
        
    def forward(self, x, adj, flags):
    
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:,_,:,:])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)) , flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid  = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

            elif _==self.num_layers-1:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)   

    def forward(self, x, adj, flags=None):

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):

            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score


class ScoreNetworkA(BaselineNetwork):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(ScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                                            c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')
        
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
            elif _==self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)  

    def forward(self, x, adj, flags,t):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):

            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)
        
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score

class ScoreNetworkA_poincare(BaselineNetwork):

    def __init__(self, max_feat_num,max_node_num, nhid, num_layers, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN',manifold=None,**kwargs):

        super(ScoreNetworkA_poincare, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears,
                                                     c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.manifold = manifold
        if self.manifold is not None:
            self.centroid = CentroidDistance(self.nfeat,self.nfeat,manifold)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init,
                                                  self.c_hid, self.num_heads, self.conv))
            elif _ == self.num_layers - 1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                  self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                  self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=1,
                         use_bn=False, activate_func=F.elu)

        self.temb_net = MLP(num_layers=3, input_dim=self.nfeat, hidden_dim=2 * self.nfeat, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)

        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.time_scale = nn.Sequential(
            nn.Linear(self.nfeat, self.nfeat),
            nn.ReLU(),
            nn.Linear(self.nfeat, 1)
        )
        self.mask.unsqueeze_(0)

    def forward(self, x, adj, flags,t,protos):
        temb = get_timestep_embedding(t, self.nfeat)
        if self.manifold is not None:
            x = self.centroid(x)

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)   # cat的dim进mlp
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape) * self.time_scale(temb)
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)
        return score

class ScoreNetworkA_poincare_proto(BaselineNetwork):

    def __init__(self, max_feat_num,max_node_num, nhid, num_layers, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN',manifold=None,**kwargs):

        super().__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears,
                                                     c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.manifold = manifold
        self.proto_weight = kwargs.get('proto_weight', 0.1)  # Default weight for proto guidance
        if self.manifold is not None:
            self.centroid = CentroidDistance(self.nfeat,self.nfeat,manifold)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init,
                                                  self.c_hid, self.num_heads, self.conv))
            elif _ == self.num_layers - 1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                  self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                  self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=1,
                         use_bn=False, activate_func=F.elu)

        self.temb_net = MLP(num_layers=3, input_dim=self.nfeat, hidden_dim=2 * self.nfeat, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)

        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.time_scale = nn.Sequential(
            nn.Linear(self.nfeat, self.nfeat),
            nn.ReLU(),
            nn.Linear(self.nfeat, 1)
        )
        self.mask.unsqueeze_(0)
        self.proto_proj = MLP(          # 3 层 MLP，和 temb_net 写法一致
            num_layers=3,
            input_dim=max_feat_num,     # 10
            hidden_dim=2*max_feat_num,  # 20
            output_dim=nhid,            # 32
            use_bn=False,
            activate_func=F.elu
        )

    def forward(self, x, adj, flags, t, labels, protos):
        temb = get_timestep_embedding(t, self.nfeat)
        if self.manifold is not None:
            x = self.centroid(x)

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for i in range(self.num_layers):
            x, adjc = self.layers[i](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)   # (B, N, N, L)
        out_shape = adjs.shape[:-1]  # (B, N, N)
        score = self.final(adjs).view(*out_shape) * self.time_scale(temb)

        # --- 💡 Proto引导 ---
        if protos is not None:
            proto = protos[labels]                       # (B, N, 10)
            proto = self.proto_proj(proto)               # (B, N, 32)
            # 计算每个节点到proto的相似性（负欧氏距离）
            proto_dist = torch.norm(x - proto, dim=-1)  # (B, N)
            proto_sim = -proto_dist

            proto_diff = proto_sim.unsqueeze(2) - proto_sim.unsqueeze(1)  # (B, N, N)
            proto_edge_affinity = -torch.abs(proto_diff)  # (B, N, N)

            score = score + self.proto_weight * proto_edge_affinity

        # --- Mask处理 ---
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)

        return score





class HScoreNetworkA(BaselineNetwork):

    def __init__(self, max_feat_num,max_node_num, nhid, num_layers, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN',c=None,manifold=None):

        super(HScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears,
                                                     c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.manifold = manifold
        if self.manifold is not None:
            self.centroid = CentroidDistance(self.nfeat,self.nfeat,manifold)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(HAttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init,
                                                  self.c_hid, self.num_heads, self.conv,self.manifold))
            elif _ == self.num_layers - 1:
                self.layers.append(HAttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                  self.c_final, self.num_heads, self.conv,self.manifold))
            else:
                self.layers.append(HAttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                  self.c_hid, self.num_heads, self.conv,self.manifold))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=1,
                         use_bn=False, activate_func=F.elu)
        self.temb_net = MLP(num_layers=3, input_dim=self.nfeat, hidden_dim=2 * self.nfeat, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)

    def forward(self, x, adj, flags,t):
        adjc = pow_tensor(adj, self.c_init)     # B x C_i x N x N
        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)

        return score