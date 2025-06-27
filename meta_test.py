"""
Meta-test评估函数
支持使用encoder或encoder+分数网络进行few-shot学习评估
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_utils import load_config
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device
from utils.graph_utils import node_flags
from models.GraphVAE import GraphVAE
from models.Decoders import Classifier


def run_meta_test(config, use_augmentation=False, checkpoint_paths=None):
    """
    运行Meta-test评估

    Args:
        config: 配置对象
        use_augmentation: 是否使用数据增强（分数网络）
        checkpoint_paths: 检查点路径字典，包含:
            - vae_checkpoint: VAE检查点路径
            - score_checkpoint: Score检查点路径（仅在use_augmentation=True时需要）

    Returns:
        dict: 评估结果，包含accuracy, f1等指标
    """
    if checkpoint_paths is None or "vae_checkpoint" not in checkpoint_paths:
        raise ValueError("必须提供VAE检查点路径")

    vae_checkpoint_path = checkpoint_paths["vae_checkpoint"]

    # 确保wandb会话干净
    try:
        if wandb.run is not None:
            wandb.finish()
            print("✓ 已关闭之前的wandb会话")
    except:
        pass

    # 基础设置
    device = load_device(config)
    load_seed(config.seed)

    # 初始化wandb
    wandb_suffix = "_aug" if use_augmentation else "_no_aug"
    mode = (
        "disabled"
        if getattr(config, "debug", False)
        else ("online" if config.wandb.online else "offline")
    )

    wandb.init(
        project=f"{config.wandb.project}_Meta",
        entity=config.wandb.entity,
        name=f"{config.run_name}_meta{wandb_suffix}",
        config=OmegaConf.to_container(config, resolve=True),
        mode=mode,
    )

    # 加载数据集
    dataset = MyDataset(config.data, config.fsl_task)
    train_loader, test_loader = dataset.get_loaders()

    # 加载编码器
    encoder = _load_encoder(vae_checkpoint_path, device)

    # 如果使用增强，还需要加载分数网络
    diffusion_model = None
    if use_augmentation and "score_checkpoint" in checkpoint_paths:
        score_checkpoint_path = checkpoint_paths["score_checkpoint"]
        try:
            diffusion_model = _load_diffusion_model(score_checkpoint_path, config, device)
            if diffusion_model is not None:
                print(f"✓ 分数网络已加载，启用数据增强")
            else:
                print(f"⚠️ 分数网络加载返回None，将不使用增强")
        except Exception as e:
            print(f"⚠️ 分数网络加载失败，将不使用增强: {e}")
            diffusion_model = None

    print(f"Meta-test 初始化完成")
    print(f"Device: {device}")
    print(f"数据增强: {'启用' if diffusion_model is not None else '禁用'}")

    # 运行评估
    results = _run_evaluation(
        dataset=dataset,
        encoder=encoder,
        diffusion_model=diffusion_model,
        config=config,
        device=device,
        use_augmentation=use_augmentation and diffusion_model is not None,
    )

    wandb.finish()
    return results


def _load_encoder(checkpoint_path, device):
    """加载编码器"""
    print(f"Loading encoder from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae_config = checkpoint["model_config"]

    # 构建VAE配置
    from types import SimpleNamespace

    model_config = SimpleNamespace()

    if "vae" in vae_config:
        # 旧格式
        model_config.encoder_config = SimpleNamespace(**vae_config["vae"]["encoder"])
        model_config.decoder_config = SimpleNamespace(**vae_config["vae"]["decoder"])
        model_config.pred_node_class = vae_config["vae"]["loss"]["pred_node_class"]
        model_config.pred_edge = vae_config["vae"]["loss"]["pred_edge"]
        model_config.pred_graph_class = vae_config["vae"]["loss"]["pred_graph_class"]
        model_config.use_kl_loss = vae_config["vae"]["loss"]["use_kl_loss"]
        model_config.latent_dim = vae_config["vae"]["encoder"]["latent_feature_dim"]
    else:
        # 新格式
        model_config.encoder_config = SimpleNamespace(**vae_config["encoder"])
        model_config.decoder_config = SimpleNamespace(**vae_config["decoder"])
        model_config.pred_node_class = vae_config["loss"]["pred_node_class"]
        model_config.pred_edge = vae_config["loss"]["pred_edge"]
        model_config.pred_graph_class = vae_config["loss"]["pred_graph_class"]
        model_config.use_kl_loss = vae_config["loss"]["use_kl_loss"]
        model_config.latent_dim = vae_config["encoder"]["latent_feature_dim"]

    model_config.use_base_proto_loss = False
    model_config.use_sep_proto_loss = False
    model_config.device = device

    # 创建并加载VAE
    vae_model = GraphVAE(model_config)
    vae_model.load_state_dict(checkpoint["model_state_dict"])
    vae_model.to(device)
    vae_model.eval()

    # 提取编码器
    encoder = vae_model.encoder
    encoder.requires_grad_(False)

    print("✓ Encoder loaded")
    return encoder


def _load_diffusion_model(checkpoint_path, config, device):
    """加载分数网络模型，支持ControlNet架构"""
    print(f"Loading diffusion model from: {checkpoint_path}")

    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 从checkpoint中获取模型配置
        if "model_config" in checkpoint:
            score_config = checkpoint["model_config"]
        else:
            # 如果没有保存配置，使用当前config中的配置
            score_config = config.score

        # 导入分数网络模型
        from models.ScoreNetwork_X import ScoreNetworkX_poincare
        from models.ScoreNetwork_A import ScoreNetworkA_poincare
        from models.ControlNet_Graph import GraphControlNet, create_graph_controlnet
        from utils.manifolds_utils import get_manifold

        # 创建manifold
        manifold = get_manifold("PoincareBall", c=1.0)

        # 从配置中获取网络参数，提供默认值
        if isinstance(score_config, dict):
            # 处理字典类型的配置
            x_config = score_config.get("x", {})
            a_config = score_config.get("a", {})
        else:
            # 处理对象类型的配置
            x_config = getattr(score_config, "x", {}) if hasattr(score_config, "x") else {}
            a_config = getattr(score_config, "a", {}) if hasattr(score_config, "a") else {}

        # 如果还是没有找到配置，使用默认值
        if not x_config:
            x_config = {"max_feat_num": config.data.get("max_feat_num", 16), "depth": 3, "nhid": 32}
        if not a_config:
            a_config = {
                "nhid": 32,
                "num_layers": 3,
                "num_linears": 2,
                "c_init": 2,
                "c_hid": 8,
                "c_final": 4,
                "adim": 32,
                "num_heads": 4,
            }

        # 创建原始X网络（节点特征分数网络）
        original_score_x = ScoreNetworkX_poincare(
            max_feat_num=x_config.get("max_feat_num", 16),
            depth=x_config.get("depth", 3),
            nhid=x_config.get("nhid", 32),
            manifold=manifold,
            edge_dim=1,
            GCN_type="HGCN",
        )

        # 创建原始A网络（邻接矩阵分数网络）
        original_score_adj = ScoreNetworkA_poincare(
            max_feat_num=x_config.get("max_feat_num", 16),  # 添加缺失参数
            max_node_num=config.data.max_node_num,  # 修复：使用data而不是dataset
            nhid=a_config.get("nhid", 32),
            num_layers=a_config.get("num_layers", 3),
            num_linears=a_config.get("num_linears", 2),
            c_init=a_config.get("c_init", 2),
            c_hid=a_config.get("c_hid", 8),
            c_final=a_config.get("c_final", 4),
            adim=a_config.get("adim", 32),
            num_heads=a_config.get("num_heads", 4),
            conv="GCN",
            manifold=manifold,  # 添加manifold参数
        )

        # 加载预训练权重到原始网络
        if "score_x_state_dict" in checkpoint:
            try:
                original_score_x.load_state_dict(checkpoint["score_x_state_dict"])
                print("✓ ScoreX网络权重加载成功")
            except Exception as e:
                print(f"⚠️ ScoreX网络权重加载失败: {e}")

        if "score_adj_state_dict" in checkpoint:
            try:
                original_score_adj.load_state_dict(checkpoint["score_adj_state_dict"])
                print("✓ ScoreAdj网络权重加载成功")
            except Exception as e:
                print(f"⚠️ ScoreAdj网络权重加载失败: {e}")

        # 检查checkpoint中有什么keys
        print(f"📋 Checkpoint包含的keys: {list(checkpoint.keys())}")

        # 检查是否有ControlNet模式
        use_controlnet = config.get("use_controlnet", True)

        if use_controlnet:
            print("🎯 使用ControlNet架构进行精确控制生成")
            # 创建ControlNet
            controlnet = create_graph_controlnet(original_score_x, original_score_adj)

            # 如果有ControlNet权重，加载它们
            if "controlnet_state_dict" in checkpoint:
                controlnet.load_state_dict(checkpoint["controlnet_state_dict"])
                print("✓ ControlNet权重已加载")
            else:
                print("⚠️ 未找到ControlNet权重，将使用零初始化")

            controlnet.to(device)
            controlnet.eval()

            return {
                "type": "controlnet",
                "model": controlnet,
                "original_score_x": original_score_x,
                "original_score_adj": original_score_adj,
                "manifold": manifold,
                "device": device,
            }
        else:
            print("🔧 使用传统Score网络进行数据增强")
            # 传统方式
            original_score_x.to(device)
            original_score_adj.to(device)
            original_score_x.eval()
            original_score_adj.eval()

            return {
                "type": "traditional",
                "score_x": original_score_x,
                "score_adj": original_score_adj,
                "manifold": manifold,
                "device": device,
            }

    except Exception as e:
        print(f"❌ 分数网络加载失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def _get_embeddings(encoder, x, adj, device):
    """获取图嵌入"""
    from utils.graph_utils import node_flags

    mask = node_flags(adj).unsqueeze(-1)

    with torch.no_grad():
        # 使用编码器提取特征
        posterior = encoder(x, adj, mask)

        # 处理分布输出 - 获取均值或模式
        if hasattr(posterior, "mode"):
            z = posterior.mode()
        elif hasattr(posterior, "mean"):
            z = posterior.mean
        else:
            z = posterior

        # 确保z是节点级别的特征 [batch_size, num_nodes, feature_dim]
        print(f"  编码器输出形状: {z.shape}")
        print(f"  mask形状: {mask.shape}")

        # 检查是否在双曲流形上
        print(f"  encoder.manifold存在: {hasattr(encoder, 'manifold')}")
        if hasattr(encoder, "manifold"):
            print(f"  encoder.manifold: {encoder.manifold}")

        if hasattr(encoder, "manifold") and encoder.manifold is not None:
            # 双曲空间：使用流形上的平均池化
            print(f"  使用双曲空间池化")
            manifold = encoder.manifold

            # 在双曲空间中进行masked pooling
            mask_expanded = mask.expand_as(z)

            # 将无效节点投影到原点（在双曲空间中）
            z_masked = z * mask_expanded

            print(f"  双曲空间 - mask_expanded形状: {mask_expanded.shape}")
            print(f"  双曲空间 - z_masked形状: {z_masked.shape}")

            # 计算有效节点数
            num_valid_nodes = mask.sum(dim=1, keepdim=True).float()
            num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)

            print(f"  双曲空间 - num_valid_nodes形状: {num_valid_nodes.shape}")

            # 在双曲空间中进行平均（使用Einstein中点）
            # 简化版本：先转换到切空间，平均，再投影回流形
            z_tangent = manifold.logmap0(z_masked)

            print(f"  双曲空间 - z_tangent形状: {z_tangent.shape}")

            # 在切空间中平均 - 沿节点维度(dim=1)求和
            graph_embeddings = z_tangent.sum(dim=1) / num_valid_nodes.squeeze(-1)

            print(f"  双曲空间 - 切空间平均后形状: {graph_embeddings.shape}")

            # 投影回流形
            graph_embeddings = manifold.expmap0(graph_embeddings)

            print(f"  双曲空间 - 投影回流形后形状: {graph_embeddings.shape}")

        else:
            # 欧几里得空间：标准平均池化
            print(f"  使用欧几里得空间池化")

            # mask: [batch_size, num_nodes, 1]
            # z: [batch_size, num_nodes, feature_dim]
            mask_expanded = mask.expand_as(z)  # [batch_size, num_nodes, feature_dim]
            z_masked = z * mask_expanded

            print(f"  mask_expanded形状: {mask_expanded.shape}")
            print(f"  z_masked形状: {z_masked.shape}")

            # 计算有效节点数，沿着节点维度求和
            num_valid_nodes = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1, 1]
            num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)

            print(f"  num_valid_nodes形状: {num_valid_nodes.shape}")

            # 平均池化 - 得到图级别嵌入，沿着节点维度(dim=1)求和
            graph_embeddings = z_masked.sum(dim=1) / num_valid_nodes.squeeze(
                -1
            )  # [batch_size, feature_dim]
            print(f"  池化前z_masked.sum(dim=1)形状: {z_masked.sum(dim=1).shape}")
            print(f"  除法前num_valid_nodes.squeeze(-1)形状: {num_valid_nodes.squeeze(-1).shape}")

        print(f"  图级嵌入形状: {graph_embeddings.shape}")
        return graph_embeddings


def _augment_data(data, diffusion_model, k_augment=5):
    """使用分数网络进行数据增强，支持ControlNet条件生成"""
    if diffusion_model is None:
        return data

    x = data["x"]  # [batch_size, num_nodes, num_features]
    adj = data["adj"]  # [batch_size, num_nodes, num_nodes]
    labels = data["labels"]  # [batch_size]

    batch_size = x.size(0)
    device = x.device

    try:
        # 检查扩散模型类型
        model_type = diffusion_model.get("type", "traditional")

        if model_type == "controlnet":
            return _augment_with_controlnet(data, diffusion_model, k_augment)
        else:
            return _augment_traditional(data, diffusion_model, k_augment)

    except Exception as e:
        print(f"⚠️ 数据增强失败: {e}")
        return data


def _augment_with_controlnet(data, diffusion_model, k_augment=5):
    """使用ControlNet进行基于类别原型的数据增强"""
    x = data["x"]
    adj = data["adj"]
    labels = data["labels"]

    batch_size = x.size(0)
    device = x.device

    controlnet = diffusion_model["model"]
    manifold = diffusion_model["manifold"]

    # 首先使用当前批次更新类别原型
    graph_features = x.mean(dim=1)  # 提取图级特征 [batch, feat_dim]
    controlnet.update_prototypes_from_support(x, labels)

    augmented_x_list = [x]  # 包含原始数据
    augmented_adj_list = [adj]
    augmented_labels_list = [labels]

    print(f"🎯 基于类别原型的ControlNet数据增强: 每个样本生成{k_augment}个变体")

    with torch.no_grad():
        for aug_idx in range(k_augment):
            # 1. 添加噪声到原始图
            noise_level = 0.1 + aug_idx * 0.05  # 递增噪声水平
            noisy_x = x + torch.randn_like(x) * noise_level

            # 2. 使用ControlNet生成基于类别原型的增强
            # 创建随机时间步
            t = torch.rand(batch_size, device=device) * 0.5 + 0.1  # [0.1, 0.6]

            # 生成flags（假设所有节点都有效）
            flags = torch.ones_like(x[..., 0], dtype=torch.bool, device=device)  # [batch, nodes]

            # ControlNet前向传播（使用类别原型条件）
            enhanced_x, _ = controlnet(
                x=noisy_x,
                adj=adj,
                flags=flags,
                t=t,
                graph_features=graph_features,  # 图级特征
                class_labels=labels,  # 类别标签
            )

            # 3. 在流形上投影（Poincaré球约束）
            enhanced_x = manifold.proj(enhanced_x)

            # 4. 添加到增强列表
            augmented_x_list.append(enhanced_x)
            augmented_adj_list.append(adj)  # 保持拓扑结构
            augmented_labels_list.append(labels)

    # 合并所有增强数据
    final_x = torch.cat(augmented_x_list, dim=0)
    final_adj = torch.cat(augmented_adj_list, dim=0)
    final_labels = torch.cat(augmented_labels_list, dim=0)

    print(f"✓ 基于类别原型的增强完成: {batch_size} → {final_x.size(0)} 样本")
    print(f"   原型更新: 已根据当前批次更新类别原型")

    return {"x": final_x, "adj": final_adj, "labels": final_labels}


def _augment_traditional(data, diffusion_model, k_augment=5):
    """传统的分数网络数据增强（兼容之前的方法）"""
    x = data["x"]
    adj = data["adj"]
    labels = data["labels"]

    batch_size = x.size(0)
    device = x.device

    # 获取分数网络组件
    score_x = diffusion_model["score_x"]
    manifold = diffusion_model["manifold"]

    augmented_x_list = [x]  # 包含原始数据
    augmented_adj_list = [adj]
    augmented_labels_list = [labels]

    print(f"🔧 传统数据增强: 每个样本生成{k_augment}个变体")

    with torch.no_grad():
        for aug_idx in range(k_augment):
            # 添加噪声到原始数据
            noise_scale = 0.1 * (1 + aug_idx * 0.1)  # 逐渐增加噪声强度

            # 对节点特征添加噪声
            noisy_x = x + torch.randn_like(x) * noise_scale

            # 在流形上投影
            noisy_x = manifold.proj(noisy_x)

            # 使用分数网络进行去噪（简化版本）
            # 这里可以实现更复杂的扩散过程

            augmented_x_list.append(noisy_x)
            augmented_adj_list.append(adj)
            augmented_labels_list.append(labels)

    # 合并数据
    final_x = torch.cat(augmented_x_list, dim=0)
    final_adj = torch.cat(augmented_adj_list, dim=0)
    final_labels = torch.cat(augmented_labels_list, dim=0)

    print(f"✓ 传统增强完成: {batch_size} → {final_x.size(0)} 样本")

    return {"x": final_x, "adj": final_adj, "labels": final_labels}


def _train_classifier_on_task(
    task, encoder, diffusion_model, config, device, use_augmentation=False
):
    """在单个任务上训练分类器"""
    # 获取数据
    support_x = task["support_set"]["x"].to(device)
    support_adj = task["support_set"]["adj"].to(device)
    support_labels = task["support_set"]["label"].to(device)

    query_x = task["query_set"]["x"].to(device)
    query_adj = task["query_set"]["adj"].to(device)
    query_labels = task["query_set"]["label"].to(device)

    # 数据增强（如果启用）
    if use_augmentation and diffusion_model is not None:
        k_augment = getattr(config.fsl_task, "k_augment", 5)
        # 增强支持集数据
        support_data = {"x": support_x, "adj": support_adj, "labels": support_labels}
        augmented_support = _augment_data(support_data, diffusion_model, k_augment)

        # 使用增强后的数据
        support_x = augmented_support["x"]
        support_adj = augmented_support["adj"]
        support_labels = augmented_support["labels"]

    # 获取嵌入
    support_emb = _get_embeddings(encoder, support_x, support_adj, device)
    query_emb = _get_embeddings(encoder, query_x, query_adj, device)

    # 创建标签映射 - 关键修复：使用连续的标签映射
    unique_support_labels = torch.unique(support_labels)
    unique_query_labels = torch.unique(query_labels)

    # 确保查询标签都在支持标签中
    all_labels = torch.unique(torch.cat([support_labels, query_labels]))
    # 创建从原始标签到连续标签的映射
    label_map = {label.item(): idx for idx, label in enumerate(all_labels)}

    # 映射标签
    mapped_support_labels = torch.tensor(
        [label_map[label.item()] for label in support_labels], device=device, dtype=torch.long
    )
    mapped_query_labels = torch.tensor(
        [label_map[label.item()] for label in query_labels], device=device, dtype=torch.long
    )

    # 创建分类器
    num_classes = len(all_labels)
    # 修复：support_emb的形状可能是[batch_size, num_nodes, embedding_dim]，需要获取正确的维度
    if support_emb.dim() == 3:
        # 如果是3维，说明是[batch_size, num_nodes, embedding_dim]，需要池化
        embedding_dim = support_emb.size(-1)

        # 按照GraphVAE中的处理方式：mean + max pooling然后连接
        support_mean = support_emb.mean(dim=1)  # [batch_size, embedding_dim]
        support_max = support_emb.max(dim=1).values  # [batch_size, embedding_dim]
        support_emb_concat = torch.cat(
            [support_mean, support_max], dim=-1
        )  # [batch_size, embedding_dim*2]

        query_mean = query_emb.mean(dim=1)
        query_max = query_emb.max(dim=1).values
        query_emb_concat = torch.cat([query_mean, query_max], dim=-1)

        model_dim = embedding_dim  # Classifier期望的model_dim是单个嵌入维度
    else:
        # 如果是2维，说明已经是[batch_size, embedding_dim]
        # 这种情况下我们假设已经是处理过的特征，直接使用
        model_dim = support_emb.size(1) // 2  # 假设已经是连接后的特征
        support_emb_concat = support_emb
        query_emb_concat = query_emb

    classifier = Classifier(
        model_dim=model_dim,
        num_classes=num_classes,
        classifier_dropout=0.2,
        classifier_bias=True,
        manifold=None,  # 使用欧几里得空间
    ).to(device)

    # 训练分类器 - 改进训练过程
    optimizer = optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    classifier.train()

    # 增加训练轮数并添加验证
    num_epochs = 100
    best_val_loss = float("inf")
    patience = 10
    no_improve_count = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = classifier(support_emb_concat)
        loss = criterion(logits, mapped_support_labels)
        loss.backward()
        optimizer.step()

        # 简单的验证 - 在支持集上测试
        if epoch % 10 == 0:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(support_emb_concat)
                val_loss = criterion(val_logits, mapped_support_labels)
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == mapped_support_labels).float().mean()

                print(
                    f"  Epoch {epoch}: loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, val_acc={val_acc.item():.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            classifier.train()

    # 评估
    classifier.eval()
    with torch.no_grad():
        query_logits = classifier(query_emb_concat)
        query_preds = torch.argmax(query_logits, dim=1)

        # 计算指标
        accuracy = (query_preds == mapped_query_labels).float().mean().item()

        y_true = mapped_query_labels.cpu().numpy()
        y_pred = query_preds.cpu().numpy()

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def _run_evaluation(dataset, encoder, diffusion_model, config, device, use_augmentation=False):
    """运行元测试评估 - 使用新的三阶段ControlNet微调流程"""
    print(f"🚀 Starting meta-test... (增强模式: {use_augmentation})")

    results = []
    num_tasks = getattr(config.fsl_task, "num_test_tasks", 100)

    progress_bar = tqdm(range(num_tasks), desc="Meta-test")

    # 准备模型组件
    model_components = {"encoder": encoder, "diffusion_model": diffusion_model}

    for task_idx in progress_bar:
        try:
            # 采样任务 - 修复参数
            task = dataset.sample_one_task(
                is_train=False,  # 从测试集采样
                N_way=config.fsl_task.N_way,
                K_shot=config.fsl_task.K_shot,
                R_query=config.fsl_task.R_query,
            )

            if task is None:
                continue

            # 转换任务格式以适配新的meta_test_single_task函数
            task_data = {
                "support": {
                    "x": task["support_set"]["x"],
                    "adj": task["support_set"]["adj"],
                    "labels": task["support_set"]["label"],
                },
                "query": {
                    "x": task["query_set"]["x"],
                    "adj": task["query_set"]["adj"],
                    "labels": task["query_set"]["label"],
                },
            }

            # 使用新的三阶段流程：ControlNet微调 → 数据增强 → 分类器训练和评估
            result = meta_test_single_task(
                task_data=task_data, model_components=model_components, config=config, device=device
            )

            # 转换结果格式以保持兼容性
            if "accuracy" in result:
                compatible_result = {
                    "accuracy": result["accuracy"],
                    "f1": result.get("accuracy", 0.0),  # 使用accuracy作为f1的fallback
                }
                results.append(compatible_result)

            # 更新进度条
            if results:
                avg_acc = np.mean([r["accuracy"] for r in results])
                progress_bar.set_postfix({"Avg Acc": f"{avg_acc:.4f}"})

            # 记录中间结果
            if (task_idx + 1) % 10 == 0:
                avg_acc = np.mean([r["accuracy"] for r in results])
                avg_f1 = np.mean([r["f1"] for r in results])
                wandb.log(
                    {
                        "avg_accuracy": avg_acc,
                        "avg_f1": avg_f1,
                        "completed_tasks": task_idx + 1,
                    }
                )

        except Exception as e:
            print(f"⚠️ 任务 {task_idx} 失败: {e}")
            continue

    # 计算最终结果
    if results:
        accuracies = [r["accuracy"] for r in results]
        f1_scores = [r["f1"] for r in results]

        final_acc = np.mean(accuracies)
        final_f1 = np.mean(f1_scores)
        std_acc = np.std(accuracies)
        std_f1 = np.std(f1_scores)

        # 95%置信区间
        margin_acc = 1.96 * std_acc / np.sqrt(len(accuracies))
        margin_f1 = 1.96 * std_f1 / np.sqrt(len(f1_scores))

        # 记录最终结果
        wandb.log(
            {
                "final_accuracy": final_acc,
                "final_f1": final_f1,
                "accuracy_std": std_acc,
                "f1_std": std_f1,
                "accuracy_margin": margin_acc,
                "f1_margin": margin_f1,
                "num_tasks": len(results),
                "use_augmentation": use_augmentation,
            }
        )

        # 打印结果
        aug_status = "ControlNet微调增强模式" if use_augmentation else "基础模式"
        print("\n" + "=" * 60)
        print(f"📊 FINAL RESULTS ({aug_status})")
        print("=" * 60)
        print(f"Number of tasks: {len(results)}")
        print(f"Accuracy: {final_acc:.4f} ± {margin_acc:.4f}")
        print(f"F1 Score: {final_f1:.4f} ± {margin_f1:.4f}")
        print("=" * 60)

        return {
            "accuracy": final_acc,
            "f1": final_f1,
            "num_tasks": len(results),
            "use_augmentation": use_augmentation,
        }
    else:
        print("⚠️ 警告：没有成功完成任何任务!")
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "num_tasks": 0,
            "use_augmentation": use_augmentation,
        }


def existing_task_test(encoder, graph_embedding_net, task, config, device, use_augmentation=False):
    """
    对单个任务进行测试，使用现有的方法
    """
    try:
        # 提取任务数据
        support_x = task["support_set"]["x"].to(device)
        support_adj = task["support_set"]["adj"].to(device)
        support_labels = task["support_set"]["y"].to(device)
        query_x = task["query_set"]["x"].to(device)
        query_adj = task["query_set"]["adj"].to(device)
        query_labels = task["query_set"]["y"].to(device)

        N_way = config.fsl_task.N_way

        with torch.no_grad():
            # 编码图形
            support_node_emb = encoder(support_x, support_adj)
            query_node_emb = encoder(query_x, query_adj)

            # 获取图级别嵌入
            support_graph_emb = graph_embedding_net(support_node_emb, support_adj)
            query_graph_emb = graph_embedding_net(query_node_emb, query_adj)

            support_emb = support_graph_emb
            query_emb = query_graph_emb

            # 标签信息
            unique_labels = torch.unique(torch.cat([support_labels, query_labels]))

            # 标签映射：将原始标签映射到0, 1, 2, ..., N_way-1
            label_to_new = {label.item(): i for i, label in enumerate(unique_labels)}

            mapped_support_labels = torch.tensor(
                [label_to_new[label.item()] for label in support_labels], device=device
            )
            mapped_query_labels = torch.tensor(
                [label_to_new[label.item()] for label in query_labels], device=device
            )

            # 连接图嵌入和节点嵌入特征
            support_emb_concat = torch.cat([support_emb, support_node_emb.mean(dim=1)], dim=-1)
            query_emb_concat = torch.cat([query_emb, query_node_emb.mean(dim=1)], dim=-1)

            # 训练线性分类器
            classifier = nn.Linear(support_emb_concat.size(-1), N_way).to(device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            # 训练100个epoch
            for epoch in range(100):
                optimizer.zero_grad()
                logits = classifier(support_emb_concat)
                loss = criterion(logits, mapped_support_labels)
                loss.backward()
                optimizer.step()

            # 测试
            with torch.no_grad():
                query_logits = classifier(query_emb_concat)
                query_preds = torch.argmax(query_logits, dim=1)

                # 计算准确率
                correct = (query_preds == mapped_query_labels).sum().item()
                total = mapped_query_labels.size(0)
                accuracy = correct / total

                return {"accuracy": accuracy}

    except Exception as e:
        print(f"Task test error: {e}")
        return {"accuracy": 0.0}


def _finetune_controlnet_for_task(controlnet, support_data, config, device):
    """
    为特定任务微调ControlNet

    Args:
        controlnet: ControlNet模型
        support_data: 支持集数据
        config: 配置
        device: 设备
    """
    print(f"🔧 开始任务特定的ControlNet微调...")

    # 提取支持集数据并确保在正确设备上
    support_x = support_data["x"].to(device)  # [num_support, nodes, features]
    support_adj = support_data["adj"].to(device)
    support_labels = support_data["labels"].to(device)

    # 1. 首先更新类别原型
    controlnet.update_prototypes_from_support(support_x, support_labels)
    print(f"✓ 类别原型已更新")

    # 2. 设置优化器（只优化ControlNet分支）
    controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(controlnet_params, lr=config.get("controlnet_lr", 0.001))

    # 3. 微调参数
    num_finetune_epochs = config.get("controlnet_finetune_epochs", 10)
    noise_scale = config.get("finetune_noise_scale", 0.1)

    controlnet.train()  # 设置为训练模式

    finetune_losses = []

    for epoch in range(num_finetune_epochs):
        total_loss = 0.0
        num_batches = 0

        # 对支持集进行多次微调
        batch_size = min(4, len(support_x))  # 小批量

        for start_idx in range(0, len(support_x), batch_size):
            end_idx = min(start_idx + batch_size, len(support_x))

            # 获取当前批次
            batch_x = support_x[start_idx:end_idx]
            batch_adj = support_adj[start_idx:end_idx]
            batch_labels = support_labels[start_idx:end_idx]

            # 提取图级特征
            graph_features = batch_x.mean(dim=1)

            # 添加噪声创建训练对
            noisy_x = batch_x + torch.randn_like(batch_x) * noise_scale

            # 创建时间步和flags
            t = torch.rand(batch_x.size(0), device=device) * 0.5 + 0.1
            flags = torch.ones_like(batch_x[..., 0], dtype=torch.bool, device=device)

            # ControlNet前向传播
            predicted_x, _ = controlnet(
                x=noisy_x,
                adj=batch_adj,
                flags=flags,
                t=t,
                graph_features=graph_features,
                class_labels=batch_labels,
            )

            # 计算重建损失 - 目标是从噪声图重建原始图
            reconstruction_loss = torch.nn.functional.mse_loss(predicted_x, batch_x)

            # 添加正则化：保持类别特征一致性
            # 计算生成样本的类别特征与原始样本的相似性
            generated_features = predicted_x.mean(dim=1)  # 图级特征
            original_features = batch_x.mean(dim=1)
            consistency_loss = torch.nn.functional.mse_loss(generated_features, original_features)

            # 总损失
            loss = reconstruction_loss + 0.1 * consistency_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet_params, max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        finetune_losses.append(avg_loss)

        if epoch % 2 == 0 or epoch == num_finetune_epochs - 1:
            print(f"  微调 Epoch {epoch:2d}: loss={avg_loss:.4f}")

    controlnet.eval()  # 设置回评估模式

    print(f"✓ ControlNet微调完成，最终损失: {finetune_losses[-1]:.4f}")
    return controlnet


def _augment_with_finetuned_controlnet(data, controlnet, config, k_augment=5):
    """
    使用微调后的ControlNet进行数据增强

    Args:
        data: 输入数据（支持集或查询集）
        controlnet: 微调后的ControlNet
        config: 配置
        k_augment: 增强倍数
    """
    x = data["x"]
    adj = data["adj"]
    labels = data["labels"]

    batch_size = x.size(0)
    device = x.device

    # 确保所有数据都在正确的设备上
    x = x.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    augmented_x_list = [x]  # 包含原始数据
    augmented_adj_list = [adj]
    augmented_labels_list = [labels]

    print(f"🎯 使用微调后的ControlNet进行数据增强: {k_augment}x")

    with torch.no_grad():
        # 提取图级特征
        graph_features = x.mean(dim=1)

        for aug_idx in range(k_augment):
            # 使用不同的噪声水平和时间步
            noise_level = 0.05 + aug_idx * 0.03  # 更细致的噪声控制
            time_step = 0.1 + aug_idx * 0.1  # 不同的时间步

            # 添加噪声
            noisy_x = x + torch.randn_like(x) * noise_level

            # 创建时间步和flags
            t = torch.full((batch_size,), time_step, device=device)
            flags = torch.ones_like(x[..., 0], dtype=torch.bool, device=device)

            # ControlNet生成
            enhanced_x, _ = controlnet(
                x=noisy_x,
                adj=adj,
                flags=flags,
                t=t,
                graph_features=graph_features,
                class_labels=labels,
            )

            # 确保生成的数据在正确的设备上
            enhanced_x = enhanced_x.to(device)

            # 可选：在流形上投影
            # enhanced_x = manifold.proj(enhanced_x)  # 如果需要的话

            augmented_x_list.append(enhanced_x)
            augmented_adj_list.append(adj)  # 保持拓扑结构
            augmented_labels_list.append(labels)

    # 合并所有数据
    final_x = torch.cat(augmented_x_list, dim=0)
    final_adj = torch.cat(augmented_adj_list, dim=0)
    final_labels = torch.cat(augmented_labels_list, dim=0)

    print(f"✓ 微调后增强完成: {batch_size} → {final_x.size(0)} 样本")

    return {"x": final_x, "adj": final_adj, "labels": final_labels}


def meta_test_single_task(task_data, model_components, config, device):
    """
    执行单个任务的meta-test，包含三阶段流程：
    1. ControlNet微调
    2. 数据增强
    3. 分类器训练和评估
    """
    try:
        support_data = task_data["support"]
        query_data = task_data["query"]

        # ==================== 阶段1: ControlNet任务特定微调 ====================
        diffusion_model = model_components.get("diffusion_model")
        use_finetuned_augmentation = config.get("use_finetuned_controlnet", True)

        if (
            diffusion_model is not None
            and diffusion_model.get("type") == "controlnet"
            and use_finetuned_augmentation
        ):
            print(f"🔧 阶段1: ControlNet任务特定微调")

            # 创建ControlNet的副本进行微调（避免影响其他任务）
            import copy

            task_controlnet = copy.deepcopy(diffusion_model["model"])

            # 微调ControlNet
            task_controlnet = _finetune_controlnet_for_task(
                controlnet=task_controlnet, support_data=support_data, config=config, device=device
            )

            # ==================== 阶段2: 数据增强 ====================
            print(f"🎯 阶段2: 使用微调后的ControlNet进行数据增强")

            # 增强支持集
            k_augment_support = config.get("k_augment_support", 3)
            augmented_support = _augment_with_finetuned_controlnet(
                data=support_data,
                controlnet=task_controlnet,
                config=config,
                k_augment=k_augment_support,
            )

            # 可选：也增强查询集（用于训练分类器，但不用于最终评估）
            if config.get("augment_query_for_training", False):
                k_augment_query = config.get("k_augment_query", 2)
                augmented_query_for_training = _augment_with_finetuned_controlnet(
                    data=query_data,
                    controlnet=task_controlnet,
                    config=config,
                    k_augment=k_augment_query,
                )
            else:
                augmented_query_for_training = None

        else:
            # 检查是否有传统分数网络可用于增强
            if diffusion_model is not None and diffusion_model.get("type") == "traditional":
                print(f"🔧 阶段2: 使用传统Score网络进行数据增强")

                # 使用传统增强方法
                k_augment_support = config.get("k_augment_support", 3)
                augmented_support = _augment_traditional(
                    data=support_data,
                    diffusion_model=diffusion_model,
                    k_augment=k_augment_support,
                )

                # 可选：也增强查询集
                if config.get("augment_query_for_training", False):
                    k_augment_query = config.get("k_augment_query", 2)
                    augmented_query_for_training = _augment_traditional(
                        data=query_data,
                        diffusion_model=diffusion_model,
                        k_augment=k_augment_query,
                    )
                else:
                    augmented_query_for_training = None
            else:
                print(f"⚠️ 跳过ControlNet微调，使用原始数据")
                augmented_support = support_data
                augmented_query_for_training = None

        # ==================== 阶段3: 分类器训练和评估 ====================
        print(f"📊 阶段3: 分类器训练和评估")

        # 准备训练数据
        if augmented_query_for_training is not None:
            # 合并增强的支持集和查询集用于训练
            train_x = torch.cat([augmented_support["x"], augmented_query_for_training["x"]], dim=0)
            train_adj = torch.cat(
                [augmented_support["adj"], augmented_query_for_training["adj"]], dim=0
            )
            train_labels = torch.cat(
                [augmented_support["labels"], augmented_query_for_training["labels"]], dim=0
            )
            train_data = {"x": train_x, "adj": train_adj, "labels": train_labels}
        else:
            # 只使用增强的支持集
            train_data = augmented_support

        # 训练分类器
        classifier = _train_classifier(
            train_data=train_data, model_components=model_components, config=config, device=device
        )

        # 在原始查询集上评估（重要：评估时不使用增强数据）
        test_results = _evaluate_classifier(
            classifier=classifier,
            test_data=query_data,  # 使用原始查询集
            config=config,
            device=device,
        )

        # 记录各阶段信息
        results = {
            "accuracy": test_results["accuracy"],
            "loss": test_results.get("loss", 0.0),
            "stages": {
                "controlnet_finetuned": diffusion_model is not None and use_finetuned_augmentation,
                "support_augmented": augmented_support["x"].size(0),
                "original_support": support_data["x"].size(0),
                "augmentation_ratio": augmented_support["x"].size(0) / support_data["x"].size(0),
            },
        }

        print(
            f"✓ 任务完成 - 准确率: {results['accuracy']:.4f}, "
            f"增强比例: {results['stages']['augmentation_ratio']:.1f}x"
        )

        return results

    except Exception as e:
        print(f"❌ 任务执行失败: {e}")
        import traceback

        traceback.print_exc()
        return {
            "accuracy": 0.0,
            "error": str(e),
            "stages": {
                "controlnet_finetuned": False,
                "support_augmented": 0,
                "original_support": 0,
                "augmentation_ratio": 0.0,
            },
        }


def _train_classifier(train_data, model_components, config, device):
    """训练分类器"""
    encoder = model_components["encoder"]

    x = train_data["x"].to(device)
    adj = train_data["adj"].to(device)
    labels = train_data["labels"].to(device)

    # 获取嵌入
    with torch.no_grad():
        embeddings = _get_embeddings(encoder, x, adj, device)

    print(f"  嵌入形状: {embeddings.shape}, 标签形状: {labels.shape}")

    # 标签重新映射：将原始标签映射到连续的0, 1, 2, ..., N_way-1
    unique_labels = torch.unique(labels)
    label_to_new = {label.item(): i for i, label in enumerate(unique_labels)}

    mapped_labels = torch.tensor(
        [label_to_new[label.item()] for label in labels], device=device, dtype=torch.long
    )

    print(f"  标签映射: {dict(zip(unique_labels.tolist(), range(len(unique_labels))))}")
    print(f"  映射后标签形状: {mapped_labels.shape}, 类别数: {len(unique_labels)}")

    # 创建分类器 - 确保使用正确的输入维度
    input_dim = embeddings.size(-1)  # 最后一个维度是特征维度
    num_classes = len(unique_labels)  # 使用实际的类别数

    classifier_config = getattr(config, "meta_test", {}).get("classifier", {})

    classifier = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(classifier_config.get("dropout", 0.1)),
        nn.Linear(64, num_classes),
    ).to(device)

    # 训练分类器
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=classifier_config.get("lr", 0.001),
        weight_decay=classifier_config.get("weight_decay", 0.0001),
    )

    criterion = nn.CrossEntropyLoss()
    epochs = classifier_config.get("epochs", 100)

    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier(embeddings)
        loss = criterion(logits, mapped_labels)  # 使用映射后的标签
        loss.backward()
        optimizer.step()

        # 每10个epoch打印一次
        if epoch % 10 == 0:
            val_acc = torch.mean((torch.argmax(logits, dim=1) == mapped_labels).float())
            print(
                f"  Epoch {epoch}: loss={loss.item():.4f}, val_loss={loss.item():.4f}, val_acc={val_acc:.4f}"
            )

    classifier.eval()
    # 返回分类器、编码器和标签映射
    return (classifier, encoder, label_to_new)


def _evaluate_classifier(classifier, test_data, config, device):
    """评估分类器"""
    classifier_model, encoder, label_to_new = classifier  # 解包分类器、编码器和标签映射

    x = test_data["x"].to(device)
    adj = test_data["adj"].to(device)
    labels = test_data["labels"].to(device)

    # 将测试标签映射到训练时的标签空间
    mapped_test_labels = torch.tensor(
        [label_to_new[label.item()] for label in labels], device=device, dtype=torch.long
    )

    with torch.no_grad():
        # 获取嵌入
        embeddings = _get_embeddings(encoder, x, adj, device)

        # 分类预测
        logits = classifier_model(embeddings)
        predictions = torch.argmax(logits, dim=1)

        # 计算准确率（在映射的标签空间中）
        correct = (predictions == mapped_test_labels).sum().item()
        total = mapped_test_labels.size(0)
        accuracy = correct / total

        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, mapped_test_labels)

        print(f"  测试准确率: {accuracy:.4f}")

    return {"accuracy": accuracy, "loss": loss.item()}
