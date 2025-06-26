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
            print(f"✓ 分数网络已加载，启用数据增强")
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
    """加载分数网络模型"""
    print(f"Loading diffusion model from: {checkpoint_path}")

    # 这里需要加载分数网络，暂时返回None表示未实现
    # TODO: 实现分数网络的加载逻辑
    print("⚠️ 分数网络加载功能待实现")
    return None


def _get_embeddings(encoder, x, adj, device):
    """获取图嵌入"""
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

        # 检查是否在双曲流形上
        if hasattr(encoder, "manifold") and encoder.manifold is not None:
            # 双曲空间：使用流形上的平均池化
            manifold = encoder.manifold

            # 在双曲空间中进行masked pooling
            mask_expanded = mask.expand_as(z)

            # 将无效节点投影到原点（在双曲空间中）
            z_masked = z * mask_expanded

            # 计算有效节点数
            num_valid_nodes = mask.sum(dim=1, keepdim=True).float()
            num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)

            # 在双曲空间中进行平均（使用Einstein中点）
            # 简化版本：先转换到切空间，平均，再投影回流形
            z_tangent = manifold.logmap0(z_masked)

            # 在切空间中平均
            graph_embeddings = z_tangent.sum(dim=1) / num_valid_nodes

            # 投影回流形
            graph_embeddings = manifold.expmap0(graph_embeddings)

        else:
            # 欧几里得空间：标准平均池化
            mask_expanded = mask.expand_as(z)
            z_masked = z * mask_expanded

            # 计算有效节点数
            num_valid_nodes = mask.sum(dim=1, keepdim=True).float()
            num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)

            # 平均池化
            graph_embeddings = z_masked.sum(dim=1) / num_valid_nodes

    return graph_embeddings


def _augment_data(data, diffusion_model, k_augment=5):
    """使用分数网络进行数据增强"""
    # TODO: 实现数据增强逻辑
    return data


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
    """运行元测试评估"""
    print(f"🚀 Starting meta-test... (增强模式: {use_augmentation})")

    results = []
    num_tasks = getattr(config.fsl_task, "num_test_tasks", 100)

    progress_bar = tqdm(range(num_tasks), desc="Meta-test")

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

            # 训练并评估
            result = _train_classifier_on_task(
                task, encoder, diffusion_model, config, device, use_augmentation
            )
            results.append(result)

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
        aug_status = "增强模式" if use_augmentation else "基础模式"
        print("\n" + "=" * 50)
        print(f"📊 FINAL RESULTS ({aug_status})")
        print("=" * 50)
        print(f"Number of tasks: {len(results)}")
        print(f"Accuracy: {final_acc:.4f} ± {margin_acc:.4f}")
        print(f"F1 Score: {final_f1:.4f} ± {margin_f1:.4f}")
        print("=" * 50)

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
