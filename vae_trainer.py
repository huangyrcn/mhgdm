"""
VAE训练函数 - 简化版本
支持双曲图自编码器训练，集成元测试监控
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_utils import load_config, save_config
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch, load_model
from utils.graph_utils import node_flags
from models.GraphVAE import GraphVAE
from models.Decoders import Classifier
import torch.nn.functional as F


class EarlyStopping:
    """早停机制 - 监控Meta-Test准确率"""

    def __init__(self, patience=5, min_delta=0.01, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def train_vae(config):
    """
    VAE训练主函数

    Args:
        config: 配置对象

    Returns:
        dict: 训练结果，包含最佳检查点路径等信息
    """
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
    _init_wandb(config)

    # 加载数据集
    dataset = MyDataset(config.data, config.fsl_task)
    train_loader, test_loader = dataset.get_loaders()

    # 初始化模型
    model, optimizer, scheduler, warmup_scheduler, use_warmup = _init_model(config, device)

    # 创建保存目录
    save_dir = _create_save_dir(config)

    tqdm.write(f"VAE训练初始化完成: {config.run_name}")
    tqdm.write(f"保存目录: {save_dir}")
    tqdm.write(f"设备: {device}")

    # Meta-test设置
    meta_test_enabled = hasattr(config, "fsl_task") and config.fsl_task is not None
    if meta_test_enabled:
        tqdm.write(
            f"✓ Meta-test enabled with {config.fsl_task.N_way}-way {config.fsl_task.K_shot}-shot"
        )
    else:
        tqdm.write("✗ Meta-test disabled: no fsl_task config")

    # 早停机制初始化
    enable_early_stopping = getattr(config.vae.train, "enable_early_stopping", True)

    if enable_early_stopping:
        early_stop_patience = getattr(config.vae.train, "early_stop_patience", 5)
        early_stop_min_delta = getattr(config.vae.train, "early_stop_min_delta", 0.01)
        early_stopping = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            mode="max",  # Meta-Test准确率越高越好
        )
        tqdm.write(
            f"✓ Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}"
        )
    else:
        early_stopping = None
        tqdm.write("✗ Early stopping disabled - full training curve will be recorded")

    # 初始化元测试相关组件
    encoder = model.encoder
    if meta_test_enabled:
        _init_meta_test_components(config, device)

    # 主训练循环
    best_test_loss = float("inf")
    best_meta_test_acc = 0.0
    best_checkpoint_path = None

    progress_bar = tqdm(
        range(config.vae.train.num_epochs),
        desc="Training",
        ncols=100,
        leave=True,
        ascii=True,
    )

    for epoch in progress_bar:
        # 训练阶段
        train_losses = _train_epoch(model, train_loader, optimizer, config, device)
        mean_train_loss = np.mean(train_losses["total"])

        # 提交训练损失到wandb
        train_log = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "train_rec_loss": np.mean(train_losses["rec"]),
            "train_kl_loss": np.mean(train_losses["kl"]),
            "train_edge_loss": np.mean(train_losses["edge"]),
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(train_log)

        # 更新学习率
        if scheduler:
            if use_warmup and epoch < getattr(config.vae.train, "warmup_epochs", 0):
                warmup_scheduler.step()
            else:
                scheduler.step()

        # 检查是否需要进行测试
        should_test = (epoch % config.vae.train.test_interval == 0) or (
            epoch == config.vae.train.num_epochs - 1
        )

        if should_test:
            # 测试阶段
            test_losses = _test_epoch(model, test_loader, config, device)
            mean_test_loss = np.mean(test_losses["total"])

            # 元测试评估
            meta_test_acc = 0.0
            if meta_test_enabled:
                meta_test_acc = _meta_test_evaluation(encoder, dataset, config, device, epoch)

            # 提交测试损失和指标到wandb
            test_log = {
                "epoch": epoch,
                "test_loss": mean_test_loss,
                "test_rec_loss": np.mean(test_losses["rec"]),
                "test_kl_loss": np.mean(test_losses["kl"]),
                "test_edge_loss": np.mean(test_losses["edge"]),
                "meta_test_accuracy": meta_test_acc,
            }
            wandb.log(test_log)

            # 检查是否需要保存最佳模型
            is_best_loss = mean_test_loss < best_test_loss
            is_best_meta_acc = meta_test_acc > best_meta_test_acc

            if is_best_loss:
                best_test_loss = mean_test_loss
                checkpoint_path = _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    mean_test_loss,
                    meta_test_acc,
                    save_dir,
                    "best_loss",
                    config,
                )
                progress_bar.write(f"✓ New best loss: {mean_test_loss:.6f}")

            if is_best_meta_acc:
                best_meta_test_acc = meta_test_acc
                best_checkpoint_path = _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    mean_test_loss,
                    meta_test_acc,
                    save_dir,
                    "best_meta_acc",
                    config,
                )
                progress_bar.write(f"✓ New best meta-acc: {meta_test_acc:.4f}")

            # 早停检查
            if enable_early_stopping and meta_test_enabled and early_stopping is not None:
                should_early_stop = early_stopping(meta_test_acc)
                if should_early_stop:
                    progress_bar.write(f"🛑 Early stopping triggered at epoch {epoch}")
                    progress_bar.write(f"   Best Meta-Test Acc: {early_stopping.best_score:.4f}")
                    progress_bar.write(
                        f"   No improvement for {early_stopping.patience} consecutive evaluations"
                    )
                    _save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        mean_test_loss,
                        meta_test_acc,
                        save_dir,
                        "early_stop",
                        config,
                    )
                    break

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "Train": f"{mean_train_loss:.6f}",
                    "Best-Meta": f"{best_meta_test_acc:.4f}",
                }
            )

            tqdm.write(
                f"Epoch {epoch}: Train={mean_train_loss:.6f}, Test={mean_test_loss:.6f}, Meta-Test Acc={meta_test_acc:.4f}"
            )
        else:
            # 非测试epoch
            progress_bar.set_postfix(
                {
                    "Train": f"{mean_train_loss:.6f}",
                    "Best-Meta": f"{best_meta_test_acc:.4f}",
                }
            )

    # 保存最终模型
    final_test_losses = _test_epoch(model, test_loader, config, device)
    final_mean_test_loss = np.mean(final_test_losses["total"])
    final_meta_test_acc = 0.0
    if meta_test_enabled:
        final_meta_test_acc = _meta_test_evaluation(
            encoder, dataset, config, device, config.vae.train.num_epochs - 1
        )

    final_checkpoint_path = _save_checkpoint(
        model,
        optimizer,
        scheduler,
        config.vae.train.num_epochs - 1,
        final_mean_test_loss,
        final_meta_test_acc,
        save_dir,
        "final",
        config,
    )

    tqdm.write(
        f"Training completed. Best test loss: {best_test_loss:.6f}, Best meta-test acc: {best_meta_test_acc:.4f}"
    )

    # 如果没有最佳meta准确率检查点，则使用最终检查点
    if best_checkpoint_path is None:
        best_checkpoint_path = final_checkpoint_path

    return {
        "save_dir": save_dir,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
        "best_test_loss": best_test_loss,
        "best_meta_test_acc": best_meta_test_acc,
    }


def _init_wandb(config):
    """初始化wandb"""
    mode = "disabled" if config.debug else ("online" if config.wandb.online else "offline")

    # 从配置中获取 wandb 输出目录
    wandb_output_dir = getattr(config.wandb, "output_dir", "logs")
    wandb_dir = os.path.join(wandb_output_dir, "wandb")

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.run_name,
        config=OmegaConf.to_container(config, resolve=True),
        mode=mode,
        dir=wandb_dir,
    )


def _init_model(config, device):
    """初始化VAE模型"""
    # 创建GraphVAE配置对象
    from types import SimpleNamespace

    vae_config = SimpleNamespace()
    vae_config.pred_node_class = config.vae.loss.pred_node_class
    vae_config.pred_edge = config.vae.loss.pred_edge
    vae_config.pred_graph_class = config.vae.loss.pred_graph_class
    vae_config.use_kl_loss = config.vae.loss.use_kl_loss
    vae_config.use_base_proto_loss = config.vae.loss.use_base_proto_loss
    vae_config.use_sep_proto_loss = config.vae.loss.use_sep_proto_loss

    # 设置编码器和解码器配置
    vae_config.encoder_config = config.vae.encoder
    vae_config.encoder_config.input_feature_dim = config.data.max_feat_num

    vae_config.decoder_config = config.vae.decoder
    vae_config.decoder_config.latent_feature_dim = config.vae.encoder.latent_feature_dim
    vae_config.decoder_config.output_feature_dim = config.data.max_feat_num

    vae_config.latent_dim = config.vae.encoder.latent_feature_dim
    vae_config.device = device

    # 创建GraphVAE
    model = GraphVAE(vae_config).to(device)

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.vae.train.lr,
        weight_decay=config.vae.train.weight_decay,
    )

    # 创建学习率调度器
    scheduler = None
    warmup_scheduler = None
    use_warmup = False

    if config.vae.train.lr_schedule:
        scheduler_type = getattr(config.vae.train, "scheduler_type", "exponential")

        if scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.vae.train.lr_decay)
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.vae.train.num_epochs,
                eta_min=config.vae.train.lr * 0.01,
            )
        elif scheduler_type == "step":
            step_size = getattr(config.vae.train, "lr_step_size", 100)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=config.vae.train.lr_decay
            )
        else:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.vae.train.lr_decay)

        # Warm-up调度器支持
        warmup_epochs = getattr(config.vae.train, "warmup_epochs", 0)
        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import LambdaLR

            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
            use_warmup = True
            tqdm.write(f"✓ Warm-up enabled: {warmup_epochs} epochs")

        tqdm.write(f"✓ LR Scheduler: {scheduler_type}, decay: {config.vae.train.lr_decay}")

    tqdm.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, optimizer, scheduler, warmup_scheduler, use_warmup


def _create_save_dir(config):
    """创建保存目录"""
    if hasattr(config.paths, "vae_save_dir"):
        save_dir = config.paths.vae_save_dir
    else:
        save_dir = os.path.join(config.paths.save_dir, config.exp_name, config.timestamp)

    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _init_meta_test_components(config, device):
    """初始化meta-test评估组件"""
    fsl_config = config.fsl_task
    N_way = fsl_config.N_way
    meta_test_tasks = getattr(fsl_config, "meta_test_tasks", 10)
    latent_dim = config.vae.encoder.latent_feature_dim

    print(f"✓ Meta-test components initialized:")
    print(f"  N-way: {N_way}, K-shot: {fsl_config.K_shot}, R-query: {fsl_config.R_query}")
    print(f"  Meta-test tasks: {meta_test_tasks}")
    print(f"  Latent dim: {latent_dim}")


def _train_epoch(model, train_loader, optimizer, config, device):
    """训练一个epoch"""
    model.train()
    losses = {"total": [], "rec": [], "kl": [], "edge": []}

    for batch in train_loader:
        x, adj, labels = load_batch(batch, device)

        optimizer.zero_grad()

        # 前向传播
        (
            rec_loss,
            kl_loss,
            edge_loss,
            base_proto_loss,
            sep_proto_loss,
            graph_classification_loss,
            acc_proto,
        ) = model(x, adj, labels)

        # 计算总损失
        total_loss = (
            config.vae.train.rec_weight * rec_loss
            + config.vae.train.kl_regularization * kl_loss
            + config.vae.train.edge_weight * edge_loss
            + config.vae.train.base_proto_weight * base_proto_loss
            + config.vae.train.sep_proto_weight * sep_proto_loss
            + config.vae.train.graph_classification_weight * graph_classification_loss
        )

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.vae.train.grad_norm)
        optimizer.step()

        # 记录损失
        losses["total"].append(total_loss.item())
        losses["rec"].append(rec_loss.item())
        losses["kl"].append(kl_loss.item())
        losses["edge"].append(edge_loss.item())

    return losses


def _test_epoch(model, test_loader, config, device):
    """测试一个epoch"""
    model.eval()
    losses = {"total": [], "rec": [], "kl": [], "edge": []}

    with torch.no_grad():
        for batch in test_loader:
            x, adj, labels = load_batch(batch, device)

            # 前向传播
            (
                rec_loss,
                kl_loss,
                edge_loss,
                base_proto_loss,
                sep_proto_loss,
                graph_classification_loss,
                acc_proto,
            ) = model(x, adj, labels)

            # 计算总损失
            total_loss = (
                config.vae.train.rec_weight * rec_loss
                + config.vae.train.kl_regularization * kl_loss
                + config.vae.train.edge_weight * edge_loss
                + config.vae.train.base_proto_weight * base_proto_loss
                + config.vae.train.sep_proto_weight * sep_proto_loss
                + config.vae.train.graph_classification_weight * graph_classification_loss
            )

            # 记录损失
            losses["total"].append(total_loss.item())
            losses["rec"].append(rec_loss.item())
            losses["kl"].append(kl_loss.item())
            losses["edge"].append(edge_loss.item())

    return losses


def _meta_test_evaluation(encoder, dataset, config, device, epoch):
    """Meta-test evaluation using linear probing"""
    # 使用配置文件中的FSL参数
    N_way = config.fsl_task.N_way
    K_shot = config.fsl_task.K_shot
    R_query = config.fsl_task.R_query
    meta_test_tasks = getattr(config.fsl_task, "meta_test_tasks", 10)

    encoder.eval()
    all_task_accuracies = []

    try:
        successful_tasks = 0
        failed_tasks = 0
        
        for task_idx in range(meta_test_tasks):
            task = dataset.sample_one_task(
                is_train=False,
                N_way=N_way,
                K_shot=K_shot,
                R_query=R_query,
            )

            if task is None:
                failed_tasks += 1
                continue
            
            successful_tasks += 1

            # 提取任务数据
            support_x = task["support_set"]["x"].to(device)
            support_adj = task["support_set"]["adj"].to(device)
            support_labels = task["support_set"]["label"].to(device)
            query_x = task["query_set"]["x"].to(device)
            query_adj = task["query_set"]["adj"].to(device)
            query_labels = task["query_set"]["label"].to(device)

            with torch.no_grad():
                # 提取特征
                support_features = _extract_features(encoder, support_x, support_adj, device)
                query_features = _extract_features(encoder, query_x, query_adj, device)

                # 使用原型网络测试
                accuracy = _test_with_prototypes(
                    encoder, support_features, support_labels, query_features, query_labels, N_way
                )
                all_task_accuracies.append(accuracy)

        if successful_tasks > 0:
            mean_accuracy = np.mean(all_task_accuracies)
            return mean_accuracy
        else:
            return 0.0

    except Exception as e:
        tqdm.write(f"Meta-test evaluation error: {e}")
        return 0.0


def _extract_features(encoder, x_batch, adj_batch, device):
    """使用编码器提取特征"""
    with torch.no_grad():
        # 生成node_mask
        node_mask = node_flags(adj_batch)
        node_mask = node_mask.unsqueeze(-1)  # 增加最后一个维度

        # 使用编码器提取特征
        posterior = encoder(x_batch, adj_batch, node_mask)
        z_mu = posterior.mode()  # 获取后验分布的模式

        # 对于图级别的分类，我们需要聚合节点特征
        # 使用平均池化，同时考虑node_mask
        node_mask_for_pooling = node_mask.squeeze(-1)  # [batch_size, num_nodes]
        masked_features = z_mu * node_mask.expand_as(z_mu)  # 应用mask

        # 计算每个图的有效节点数
        num_valid_nodes = node_mask_for_pooling.sum(dim=1, keepdim=True)  # [batch_size, 1]
        num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)  # 避免除零

        # 平均池化得到图级特征
        graph_features = masked_features.sum(dim=1) / num_valid_nodes  # [batch_size, latent_dim]

        return graph_features


def _test_with_prototypes(
    encoder, support_features, support_labels, query_features, query_labels, n_way
):
    """使用原型网络进行分类，支持正确的几何距离计算"""
    with torch.no_grad():
        prototypes = []
        for c in range(n_way):
            # 筛选出属于类别c的样本特征
            class_features = support_features[support_labels == c]
            # 计算原型（类别特征的均值）
            if class_features.size(0) > 0:
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)

        # 如果某个类别在支持集中没有样本，则无法评估
        if len(prototypes) != n_way:
            return 0.0

        prototypes = torch.stack(prototypes)  # [n_way, latent_dim]

        # 根据编码器类型选择正确的距离计算方式
        if hasattr(encoder, "manifold") and encoder.manifold is not None:
            # 双曲流形：使用双曲距离
            manifold = encoder.manifold

            # 扩展维度以计算批量距离
            query_expanded = query_features.unsqueeze(1)  # [num_query, 1, latent_dim]
            prototypes_expanded = prototypes.unsqueeze(0)  # [1, n_way, latent_dim]

            # 计算双曲距离
            distances = manifold.dist(query_expanded, prototypes_expanded)  # [num_query, n_way]
            distances = distances.squeeze(-1) if distances.dim() > 2 else distances
        else:
            # 欧几里得空间：使用欧几里得距离
            distances = torch.sum(
                (query_features.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2, dim=2
            )

        # 预测类别为距离最小的原型对应的类别
        predictions = torch.argmin(distances, dim=1)

        # 计算准确率
        correct = (predictions == query_labels).float().sum().item()
        accuracy = correct / len(query_labels) if len(query_labels) > 0 else 0.0

    return accuracy


def _save_checkpoint(
    model, optimizer, scheduler, epoch, test_loss, meta_test_acc, save_dir, checkpoint_type, config
):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_config": OmegaConf.to_container(config, resolve=True),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_loss": test_loss,
        "meta_test_acc": meta_test_acc,
    }

    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # 保存指定类型的检查点
    checkpoint_path = os.path.join(save_dir, f"{checkpoint_type}.pth")
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path
