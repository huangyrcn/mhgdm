"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""

# 包含了 自编码器的注意力机制部分
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout, edge_dim=1):
        super(DenseAtt, self).__init__()
        self.edge_dim = edge_dim  # Expected dimension of edge features for the MLP
        self.att_mlp = nn.Linear(2 * in_features + self.edge_dim, 1, bias=True)
        self.dropout = dropout
        self.softmax = nn.Softmax(
            dim=-1
        )  # Softmax over columns (axis -1 of N_i x N_j attention matrix)

    def forward(self, x_left, x_right, adj, edge_attr=None):
        # x_left, x_right: (B, N, N, D_node_feat)
        # adj: (B, N, N) or (N, N) - graph structure
        # edge_attr: (B, N, N, D_actual_edge_feat) or None - explicit edge features

        # Ensure adj is suitable for mask, handling potential broadcasting
        adj_for_mask = adj
        if adj_for_mask.ndim == 2 and x_left.ndim == 4:  # adj is (N,N), x_left is (B,N,N,D)
            # This broadcasting is implicitly handled by PyTorch ops like >
            pass

        edge_mask = (adj_for_mask > 1e-5).float()  # (B,N,N) or (N,N)

        tensors_to_concat = [x_left, x_right]

        if self.edge_dim > 0:  # MLP expects edge features
            edge_features_for_mlp = None
            if edge_attr is not None:
                # Use provided edge_attr
                # Ensure its last dimension matches self.edge_dim and it's 4D
                current_edge_attr = edge_attr
                if current_edge_attr.ndim == 3 and x_left.ndim == 4:
                    # Try to make it (B,N,N,1) if shapes match and self.edge_dim is 1
                    if (
                        self.edge_dim == 1
                        and current_edge_attr.shape[0] == x_left.shape[0]
                        and current_edge_attr.shape[1] == x_left.shape[1]
                        and current_edge_attr.shape[2] == x_left.shape[2]
                    ):
                        current_edge_attr = current_edge_attr.unsqueeze(-1)

                if current_edge_attr.shape[-1] != self.edge_dim:
                    # This is a mismatch. For now, we'll raise an error if it's not correctable.
                    # Or one might add a projection layer if this is intended.
                    raise ValueError(
                        f"DenseAtt: Provided edge_attr last dimension ({current_edge_attr.shape[-1]}) "
                        f"does not match expected edge_dim ({self.edge_dim})."
                    )

                if current_edge_attr.ndim != x_left.ndim:
                    raise ValueError(
                        f"DenseAtt: Provided edge_attr ndim ({current_edge_attr.ndim}) "
                        f"does not match x_left ndim ({x_left.ndim}) after potential unsqueeze."
                    )
                edge_features_for_mlp = current_edge_attr

            else:  # edge_attr is None, MLP expects features, so derive from adj
                adj_for_features = adj  # Use the original adj passed
                if adj_for_features.ndim == 2 and x_left.ndim == 4:  # adj is (N,N)
                    # Expand to batch dim of x_left
                    adj_for_features = adj_for_features.unsqueeze(0).expand(
                        x_left.shape[0], -1, -1
                    )  # (B,N,N)
                elif (
                    adj_for_features.ndim == 3
                    and adj_for_features.shape[0] != x_left.shape[0]
                    and x_left.shape[0] != 1
                    and adj_for_features.shape[0] == 1
                ):
                    # adj is (1,N,N) and x_left is (B,N,N,D)
                    adj_for_features = adj_for_features.expand(x_left.shape[0], -1, -1)

                if self.edge_dim == 1:
                    adj_as_features = adj_for_features.unsqueeze(-1)  # (B,N,N,1)
                    edge_features_for_mlp = adj_as_features
                else:
                    # Cannot form self.edge_dim features from adj if self.edge_dim > 1
                    raise ValueError(
                        f"DenseAtt: edge_attr is None, MLP expects edge_dim={self.edge_dim} > 1. "
                        "Cannot form these features from adj."
                    )

            tensors_to_concat.append(edge_features_for_mlp)

        x_cat = torch.concat(tensors_to_concat, dim=-1)

        # Squeeze the output of MLP, which is (B,N,N,1) -> (B,N,N)
        att = self.att_mlp(x_cat).squeeze(-1)

        # Apply padding mask
        if edge_mask.ndim == 2 and att.ndim == 3:  # edge_mask (N,N), att (B,N,N)
            pad_mask = (1 - edge_mask.unsqueeze(0)).bool()  # (1,N,N) -> broadcast
        elif edge_mask.ndim == att.ndim:
            pad_mask = (1 - edge_mask).bool()
        else:
            raise ValueError(
                f"Edge mask ndim {edge_mask.ndim} not compatible with att ndim {att.ndim}"
            )

        # Using masked_fill for numerical stability with softmax
        att.masked_fill_(pad_mask, -float("inf"))

        att = self.softmax(att)  # Softmax applied on the last dimension of att (N_j)
        att = F.dropout(att, self.dropout, training=self.training)

        # Return attention coefficients, unsqueezed to be (B,N,N,1) for broadcasting with features later
        return att.unsqueeze(-1)


# TODO centroid attention


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return self.act(h_prime)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.attentions = [
            SpGraphAttentionLayer(
                input_dim, output_dim, dropout=dropout, alpha=alpha, activation=activation
            )
            for _ in range(nheads)
        ]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat(
                [att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2
            )
            h = torch.mean(h_cat, dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)
