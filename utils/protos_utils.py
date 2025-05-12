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
    unique_labels = torch.unique(all_labels)
    protos = []
    for label in unique_labels:
        mask = (all_labels == label)
        if mask.sum() > 0:
            proto = all_embeddings[mask].mean(dim=0, keepdim=True)
            protos.append(proto)
    return torch.cat(protos, dim=0).to(device)
