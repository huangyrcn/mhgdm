import torch
from utils.graph_utils import node_flags


def compute_protos_from(encoder, loader, device):
    encoder.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            # 适配数据加载器格式：(x, adj, labels)
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x, adj, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                # 如果是PyTorch Geometric格式
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                labels = batch.y.to(device)
                adj = edge_index  # 需要转换为邻接矩阵格式

            flags = node_flags(adj)
            posterior = encoder(x, adj, flags)

            # 处理编码器输出
            if hasattr(posterior, "mode"):
                embeddings = posterior.mode()
            else:
                embeddings = posterior

            # 图级别嵌入：对节点嵌入进行池化
            if len(embeddings.shape) == 3:  # [batch, nodes, features]
                # 使用平均池化得到图级别表示
                graph_embeddings = embeddings.mean(dim=1)  # [batch, features]
            else:
                graph_embeddings = embeddings

            all_embeddings.append(graph_embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    unique_labels = torch.unique(all_labels)
    protos = []
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() > 0:
            proto = all_embeddings[mask].mean(dim=0, keepdim=True)
            protos.append(proto)

    if protos:
        return torch.cat(protos, dim=0).to(device)
    else:
        # 返回默认原型
        return torch.zeros(1, all_embeddings.shape[-1]).to(device)
