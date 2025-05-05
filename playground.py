import os
import torch
import numpy as np
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # Use a default font that supports English
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import ml_collections
from easydict import EasyDict as edict

from models.HVAE import HVAE
from utils.data_utils import load_data
from utils.loader import load_batch
from utils.graph_utils import node_flags

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.loader import load_device


def visualize_encoder_output(config, checkpoint_path, save_dir="."):
    """Load a pretrained AE model, extract encoder outputs for test data,
    and visualize them using t-SNE."""

    device = load_device(config)
    seed = config.seed

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model config from checkpoint, fallback to current config if needed
    if "model_config" in checkpoint:
        if isinstance(checkpoint["model_config"], dict):
            AE_config = ml_collections.ConfigDict(checkpoint["model_config"])
        else:
            AE_config = checkpoint["model_config"]
    else:
        print("Warning: 'model_config' not found in checkpoint. Using current config.")
        AE_config = config

    AE_config.model.dropout = 0
    if "device" not in AE_config:
        AE_config.device = device
    if "data" not in AE_config:
        print(
            "Warning: Data config not found in checkpoint config, using current config data settings."
        )
        AE_config.data = config.data
    elif "train_class_num" not in AE_config.data:
        print(
            "Warning: train_class_num not found in checkpoint data config, using current config value."
        )
        if hasattr(config, "data") and hasattr(config.data, "train_class_num"):
            AE_config.data.train_class_num = config.data.train_class_num
        else:
            raise ValueError(
                "Cannot determine train_class_num. Please ensure it is in the checkpoint or config."
            )

    model = HVAE(AE_config)
    model.load_state_dict(checkpoint["ae_state_dict"], strict=False)
    model.eval()
    encoder = model.encoder.to(device)
    print("Encoder loaded.")

    print(f"Loading test data, dataset: {config.data.name}")
    _, test_loader = load_data(config)
    print("Test data loaded.")

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, adj, labels = load_batch(batch, device)
            node_mask = node_flags(adj).unsqueeze(-1)
            posterior = encoder(x, adj, node_mask)
            latents = posterior.mode()
            if latents.dim() == 3:
                graph_latents = latents.mean(dim=1)
            else:
                graph_latents = latents

            # If manifold exists and has logmap0 (e.g. PoincareBall), map to Euclidean space
            if hasattr(posterior, "manifold") and posterior.manifold is not None:
                if hasattr(posterior.manifold, "logmap0"):
                    graph_latents = posterior.manifold.logmap0(graph_latents)

            all_latents.append(graph_latents.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Latent shape: {all_latents.shape}")
    print(f"Labels shape: {all_labels.shape}")

    n_samples = all_latents.shape[0]
    perplexity_value = min(30, n_samples - 1)
    if perplexity_value <= 0:
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE. Skipping visualization.")
        return

    print(f"Applying t-SNE, perplexity={perplexity_value}...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_value, n_iter=300)
    latents_2d = tsne.fit_transform(all_latents)
    print("t-SNE done.")

    # 计算聚类指标（用降维前的 all_latents 更有意义，也可用 latents_2d）
    sil_score = silhouette_score(all_latents, all_labels)
    ch_score = calinski_harabasz_score(all_latents, all_labels)
    db_score = davies_bouldin_score(all_latents, all_labels)

    # 可选：也可以用 t-SNE 降维后的 latents_2d 计算
    # sil_score = silhouette_score(latents_2d, all_labels)
    # ch_score = calinski_harabasz_score(latents_2d, all_labels)
    # db_score = davies_bouldin_score(latents_2d, all_labels)

    # 在图上显示
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1], c=all_labels, cmap="viridis", alpha=0.7, s=10
    )
    plt.title(
        f"t-SNE of Encoder Embeddings ({config.data.name} Test Set)\n"
        f"Silhouette: {sil_score:.3f}  Calinski-Harabasz: {ch_score:.1f}  Davies-Bouldin: {db_score:.3f}"
    )
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    unique_labels = np.unique(all_labels)
    if len(unique_labels) <= 10:
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
        plt.legend(handles, unique_labels, title="Class")
    else:
        cbar = plt.colorbar(scatter)
        cbar.set_label("Class Label")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config.run_name}_tsne_visualization.png")
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE plot saved to: {save_path}")
    plt.close()


@hydra.main(config_path="configs", config_name="visual_ae", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = ml_collections.ConfigDict(cfg_dict)

    running_dir = os.path.join("configs", "running")
    os.makedirs(running_dir, exist_ok=True)

    config_path = os.path.join(running_dir, f"{cfg.run_name}.yaml")
    OmegaConf.save(cfg, config_path, resolve=True)
    print(f"Configuration saved to {config_path}")
    visualize_encoder_output(cfg_dict, cfg.ckpt_path, save_dir=cfg.save_dir)


if __name__ == "__main__":
    main()
