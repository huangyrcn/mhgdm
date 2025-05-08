import torch
from utils.graph_utils import node_flags
from utils.loader import load_batch

def compute_protos_from(encoder, loader, device):
    encoder.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, adj, labels = load_batch(batch, device)
            flags = node_flags(adj)
            posterior = encoder(x, adj, flags)
            embeddings = posterior.mode()
            all_embeddings.append(embeddings)
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 找到最大标签值，确保我们覆盖所有可能的标签
    max_label = all_labels.max().item()
    
    # 创建足够大小的空间来存储所有可能的原型
    protos = []
    
    # 对每个可能的标签值（0到max_label）创建原型
    for label_idx in range(max_label + 1):
        mask = (all_labels == label_idx)
        if mask.sum() > 0:
            # 如果有这个标签的样本，计算原型
            proto = all_embeddings[mask].mean(dim=0, keepdim=True)
        else:
            # 如果没有这个标签的样本，创建一个零向量占位
            # 或者可以用其他方式处理缺失标签
            shape = all_embeddings.shape[1:]
            proto = torch.zeros(1, *shape, device=device)
        protos.append(proto)
    
    return torch.cat(protos, dim=0).to(device)
