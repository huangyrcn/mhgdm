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
from types import SimpleNamespace

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
    """主函数 - 直观的训练流程"""
    # 确保wandb会话干净
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

    # === 准备阶段 ===
    trainer_state = setup_training(config)

    # === 训练循环 ===
    progress_bar = tqdm(
        range(config.vae.train.num_epochs),
        desc="VAE Training",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    )

    for epoch in progress_bar:
        # 训练一个epoch
        train_metrics = train_one_epoch(trainer_state, epoch)

        # 更新进度条
        progress_bar.set_postfix({"Train_Loss": f"{train_metrics['train_loss']:.3f}"})

        # 是否需要评估？
        if (epoch % config.vae.train.test_interval == 0) or (
            epoch == config.vae.train.num_epochs - 1
        ):
            eval_metrics = eval_one_epoch(trainer_state, epoch)

            # 保存最佳模型
            if eval_metrics["test_loss"] < trainer_state.best_metrics["test_loss"]:
                trainer_state.best_metrics["test_loss"] = eval_metrics["test_loss"]
                _save_checkpoint(
                    trainer_state.model,
                    trainer_state.optimizer,
                    trainer_state.scheduler,
                    epoch,
                    eval_metrics["test_loss"],
                    eval_metrics["test_meta_test_accuracy"],
                    trainer_state.save_dir,
                    "best_loss",
                    trainer_state.config,
                )

            if (
                eval_metrics["test_meta_test_accuracy"]
                > trainer_state.best_metrics["meta_test_acc"]
            ):
                trainer_state.best_metrics["meta_test_acc"] = eval_metrics[
                    "test_meta_test_accuracy"
                ]
                trainer_state.best_checkpoint_path = _save_checkpoint(
                    trainer_state.model,
                    trainer_state.optimizer,
                    trainer_state.scheduler,
                    epoch,
                    eval_metrics["test_loss"],
                    eval_metrics["test_meta_test_accuracy"],
                    trainer_state.save_dir,
                    "best_meta_acc",
                    trainer_state.config,
                )

            # 早停检查
            if (
                trainer_state.early_stopping
                and hasattr(trainer_state.config, "fsl_task")
                and trainer_state.config.fsl_task is not None
            ):

                should_stop = trainer_state.early_stopping(eval_metrics["test_meta_test_accuracy"])
                if should_stop:
                    _save_checkpoint(
                        trainer_state.model,
                        trainer_state.optimizer,
                        trainer_state.scheduler,
                        epoch,
                        eval_metrics["test_loss"],
                        eval_metrics["test_meta_test_accuracy"],
                        trainer_state.save_dir,
                        "early_stop",
                        trainer_state.config,
                    )
                    break

    progress_bar.close()

    # 保存最终模型
    final_checkpoint_path = _save_checkpoint(
        trainer_state.model,
        trainer_state.optimizer,
        trainer_state.scheduler,
        config.vae.train.num_epochs - 1,
        0.0,
        0.0,
        trainer_state.save_dir,
        "final",
        trainer_state.config,
    )

    # === 结束收尾 ===

    # 如果没有最佳meta准确率检查点，则使用最终检查点
    if (
        not hasattr(trainer_state, "best_checkpoint_path")
        or trainer_state.best_checkpoint_path is None
    ):
        trainer_state.best_checkpoint_path = final_checkpoint_path

    return {
        "save_dir": trainer_state.save_dir,
        "best_checkpoint": trainer_state.best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
        "best_test_loss": trainer_state.best_metrics["test_loss"],
        "best_meta_test_acc": trainer_state.best_metrics["meta_test_acc"],
    }


def setup_training(config):
    """一次性准备好所有东西"""

    # 基础设置
    device = load_device(config)
    load_seed(config.seed)

    # 初始化wandb
    _init_wandb(config)

    # 数据和模型
    dataset = MyDataset(config.data, config.fsl_task)
    model, optimizer, scheduler, warmup_scheduler, use_warmup = _init_model(config, device)

    # 保存和日志
    save_dir = _create_save_dir(config)

    # Meta-test设置
    meta_test_enabled = hasattr(config, "fsl_task") and config.fsl_task is not None
    if meta_test_enabled:
        _init_meta_test_components(config, device)

    # 早停和记录
    enable_early_stopping = getattr(config.vae.train, "enable_early_stopping", True)
    early_stopping = None

    if enable_early_stopping:
        early_stop_patience = getattr(config.vae.train, "early_stop_patience", 5)
        early_stop_min_delta = getattr(config.vae.train, "early_stop_min_delta", 0.01)
        early_stopping = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            mode="max",  # Meta-Test准确率越高越好
        )

    best_metrics = {"test_loss": float("inf"), "meta_test_acc": 0.0}
    best_checkpoint_path = None

    return SimpleNamespace(
        config=config,
        device=device,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        use_warmup=use_warmup,
        save_dir=save_dir,
        early_stopping=early_stopping,
        best_metrics=best_metrics,
        best_checkpoint_path=best_checkpoint_path,
    )


def train_one_epoch(state, epoch):
    """训练一个epoch"""
    model, optimizer = state.model, state.optimizer
    train_loader = state.dataset.get_loaders()[0]

    model.train()
    losses = {"total": [], "rec": [], "kl": [], "edge": []}

    for batch in train_loader:
        x, adj, labels = load_batch(batch, state.device)

        optimizer.zero_grad()

        # VAE前向传播
        (
            rec_loss,
            kl_loss,
            edge_loss,
            base_proto_loss,
            sep_proto_loss,
            graph_classification_loss,
            _,
        ) = model(x, adj, labels)

        # 总损失计算
        total_loss = (
            state.config.vae.train.rec_weight * rec_loss
            + state.config.vae.train.kl_regularization * kl_loss
            + state.config.vae.train.edge_weight * edge_loss
            + state.config.vae.train.base_proto_weight * base_proto_loss
            + state.config.vae.train.sep_proto_weight * sep_proto_loss
            + state.config.vae.train.graph_classification_weight * graph_classification_loss
        )

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), state.config.vae.train.grad_norm)
        optimizer.step()

        # 记录损失
        losses["total"].append(total_loss.item())
        losses["rec"].append(rec_loss.item())
        losses["kl"].append(kl_loss.item())
        losses["edge"].append(edge_loss.item())

    # 学习率更新
    if state.scheduler:
        if state.use_warmup and epoch < getattr(state.config.vae.train, "warmup_epochs", 0):
            state.warmup_scheduler.step()
        else:
            state.scheduler.step()

    # 记录和打印训练指标
    train_metrics = {
        "train_loss": np.mean(losses["total"]),
        "train_rec_loss": np.mean(losses["rec"]),
        "train_kl_loss": np.mean(losses["kl"]),
        "train_edge_loss": np.mean(losses["edge"]),
        "lr": optimizer.param_groups[0]["lr"],
    }

    wandb.log({**train_metrics, "epoch": epoch})

    return train_metrics


def eval_one_epoch(state, epoch):
    """评估一个epoch - 包含测试损失和元学习评估"""

    # 1. 常规测试损失
    model = state.model
    test_loader = state.dataset.get_loaders()[1]

    model.eval()
    test_losses = {"total": [], "rec": [], "kl": [], "edge": []}

    with torch.no_grad():
        for batch in test_loader:
            x, adj, labels = load_batch(batch, state.device)

            # 前向传播（和训练时相同的损失计算）
            (
                rec_loss,
                kl_loss,
                edge_loss,
                base_proto_loss,
                sep_proto_loss,
                graph_classification_loss,
                _,
            ) = model(x, adj, labels)

            total_loss = (
                state.config.vae.train.rec_weight * rec_loss
                + state.config.vae.train.kl_regularization * kl_loss
                + state.config.vae.train.edge_weight * edge_loss
                + state.config.vae.train.base_proto_weight * base_proto_loss
                + state.config.vae.train.sep_proto_weight * sep_proto_loss
                + state.config.vae.train.graph_classification_weight * graph_classification_loss
            )

            test_losses["total"].append(total_loss.item())
            test_losses["rec"].append(rec_loss.item())
            test_losses["kl"].append(kl_loss.item())
            test_losses["edge"].append(edge_loss.item())

    test_loss = np.mean(test_losses["total"])

    # 2. 元学习测试（如果启用）
    train_meta_acc, test_meta_acc = 0.0, 0.0
    if hasattr(state.config, "fsl_task") and state.config.fsl_task is not None:
        encoder = state.model.encoder
        train_meta_acc = meta_eval(
            encoder, state.dataset, state.config, state.device, is_train=True
        )
        test_meta_acc = meta_eval(
            encoder, state.dataset, state.config, state.device, is_train=False
        )

    # 记录指标
    eval_metrics = {
        "test_loss": test_loss,
        "test_rec_loss": np.mean(test_losses["rec"]),
        "test_kl_loss": np.mean(test_losses["kl"]),
        "test_edge_loss": np.mean(test_losses["edge"]),
        "train_meta_test_accuracy": train_meta_acc,
        "test_meta_test_accuracy": test_meta_acc,
    }

    wandb.log({**eval_metrics, "epoch": epoch})

    tqdm.write(
        f"Evaluation completed: Loss={test_loss:.4f}, Train_Acc={train_meta_acc:.4f}, Test_Acc={test_meta_acc:.4f}"
    )

    return eval_metrics


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
    pass


def evaluate_single_task(encoder, task, config, device):
    """
    评估单个任务的准确率 - 使用原型网络方法

    Args:
        encoder: 编码器模型
        task: 单个FSL任务，包含support_set和query_set
        config: 配置对象
        device: 设备

    Returns:
        float: 任务准确率
    """
    encoder.eval()
    N_way = config.fsl_task.N_way

    try:
        # 提取任务数据
        support_x = task["support_set"]["x"].to(device)
        support_adj = task["support_set"]["adj"].to(device)
        support_labels = task["support_set"]["label"].to(device)
        query_x = task["query_set"]["x"].to(device)
        query_adj = task["query_set"]["adj"].to(device)
        query_labels = task["query_set"]["label"].to(device)

        # 使用编码器提取特征（冻结编码器）
        with torch.no_grad():
            support_features = _extract_features(encoder, support_x, support_adj, device)
            query_features = _extract_features(encoder, query_x, query_adj, device)

        # 使用原型网络测试（默认方法）
        accuracy = _test_with_prototypical_networks(
            support_features, support_labels, query_features, query_labels, N_way, device
        )

        # 备选：线性探针方法（如果需要可以切换）
        # accuracy = _test_with_linear_probe(
        #     support_features, support_labels, query_features, query_labels, N_way, device
        # )

        return accuracy

    except Exception as e:
        tqdm.write(f"Single task evaluation error: {e}")
        return 0.0


def meta_eval(encoder, dataset, config, device, is_train=False):
    """
    元评估函数 - 在训练集或测试集上进行元学习评估

    Args:
        encoder: 编码器模型
        dataset: 数据集对象
        config: 配置对象
        device: 设备
        is_train: 是否在训练集上评估

    Returns:
        float: 平均准确率
    """
    # 使用配置文件中的FSL参数
    N_way = config.fsl_task.N_way
    K_shot = config.fsl_task.K_shot
    R_query = config.fsl_task.R_query

    encoder.eval()
    all_task_accuracies = []

    try:
        successful_tasks = 0
        failed_tasks = 0

        # 设置任务数量上限，避免训练集无限循环
        if is_train:
            # 训练集模式：限制任务数量，避免无限循环
            max_tasks = getattr(config.fsl_task, "max_train_meta_tasks", 100)
        else:
            # 测试集模式：用尽所有数据
            max_tasks = float("inf")

        while successful_tasks < max_tasks:
            # 获取任务
            if is_train:
                # 训练集：随机采样，不需要query_pool_start_index
                task = dataset.sample_one_task(
                    is_train=is_train,
                    N_way=N_way,
                    K_shot=K_shot,
                    R_query=R_query,
                )
            else:
                # 测试集：按顺序采样，传入正确的任务起始索引
                query_start_index = successful_tasks * N_way * R_query
                task = dataset.sample_one_task(
                    is_train=is_train,
                    N_way=N_way,
                    K_shot=K_shot,
                    R_query=R_query,
                    query_pool_start_index=query_start_index,
                )

            if task is None:
                # 无法采样到完整任务，停止评估
                failed_tasks += 1
                break

            # 评估单个任务
            accuracy = evaluate_single_task(encoder, task, config, device)
            all_task_accuracies.append(accuracy)
            successful_tasks += 1

        if successful_tasks > 0:
            mean_accuracy = np.mean(all_task_accuracies)
            mode = "训练集" if is_train else "测试集"
            tqdm.write(f"  {mode}: 成功任务数={successful_tasks}, 平均准确率={mean_accuracy:.4f}")
            return mean_accuracy
        else:
            return 0.0

    except Exception as e:
        tqdm.write(f"Meta-test evaluation error: {e}")
        return 0.0


def _extract_features(encoder, x_batch, adj_batch, device):
    """使用编码器提取图级特征 - 复用VAE内部的完整实现"""
    with torch.no_grad():
        # 生成node_mask
        node_mask = node_flags(adj_batch)
        node_mask = node_mask.unsqueeze(-1)  # 增加最后一个维度

        # 使用编码器提取特征
        posterior = encoder(x_batch, adj_batch, node_mask)
        emb_from_posterior = posterior.mode()  # 获取后验分布的模式

        # 使用与GraphVAE相同的图级特征提取逻辑
        if emb_from_posterior.dim() == 2:
            emb_for_pooling = emb_from_posterior.unsqueeze(1)
        else:
            emb_for_pooling = emb_from_posterior

        # 处理双曲空间映射
        if encoder.manifold is not None:
            emb_in_tangent_space = encoder.manifold.logmap0(emb_for_pooling)
        else:
            emb_in_tangent_space = emb_for_pooling

        # 应用node_mask
        masked_emb = emb_in_tangent_space * node_mask

        # 计算有效节点数
        num_valid_nodes = node_mask.sum(dim=1, keepdim=True)  # [batch_size, 1, 1]
        num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)

        # Mean + Max pooling (与GraphVAE一致)
        mean_pooled_features = masked_emb.sum(dim=1) / num_valid_nodes.squeeze(
            -1
        )  # [batch_size, latent_dim]
        max_pooled_features = masked_emb.max(dim=1).values  # [batch_size, latent_dim]

        # 拼接得到更丰富的图级表示
        graph_features = torch.cat(
            [mean_pooled_features, max_pooled_features], dim=-1
        )  # [batch_size, latent_dim*2]

        return graph_features


class LinearProbe(nn.Module):
    """线性探针分类器"""

    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def EuclideanDistances(a, b):
    """计算欧氏距离矩阵"""
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


def _test_with_prototypical_networks(
    support_features, support_labels, query_features, query_labels, n_way, device
):
    """
    使用原型网络进行分类 - 参考G-Meta代码实现

    Args:
        support_features: 支持集特征 [support_size, feature_dim]
        support_labels: 支持集标签 [support_size]
        query_features: 查询集特征 [query_size, feature_dim]
        query_labels: 查询集标签 [query_size]
        n_way: 分类类别数
        device: 设备

    Returns:
        float: 准确率
    """
    try:
        # 特征标准化
        support_features = F.normalize(support_features, p=2, dim=1)
        query_features = F.normalize(query_features, p=2, dim=1)

        # 重塑特征：[n_way, k_shot, feature_dim]
        k_shot = support_features.shape[0] // n_way
        query_size = query_features.shape[0] // n_way

        support_embs = support_features.reshape([n_way, k_shot, -1])
        query_embs = query_features.reshape([n_way, query_size, -1])

        # 计算每个类别的原型（支持集特征的平均值）
        support_protos = support_embs.mean(1)  # [n_way, feature_dim]

        # 计算查询样本与原型之间的欧氏距离的负数作为分数
        scores = -EuclideanDistances(
            query_embs.reshape([n_way * query_size, -1]), support_protos
        )  # [n_way * query_size, n_way]

        # 预测
        y_preds = torch.argmax(scores, dim=1)

        # 创建真实标签
        labels = torch.arange(n_way).unsqueeze(1).repeat(1, query_size).flatten().to(device)

        # 计算准确率
        correct = (y_preds == labels).float().sum().item()
        accuracy = correct / len(labels) if len(labels) > 0 else 0.0

        return accuracy

    except Exception as e:
        tqdm.write(f"Prototypical networks error: {e}")
        return 0.0


def _test_with_linear_probe(
    support_features, support_labels, query_features, query_labels, n_way, device
):
    """
    使用线性探针进行分类 - 简化版本

    Args:
        support_features: 支持集特征 [support_size, feature_dim]
        support_labels: 支持集标签 [support_size]
        query_features: 查询集特征 [query_size, feature_dim]
        query_labels: 查询集标签 [query_size]
        n_way: 分类类别数
        device: 设备

    Returns:
        float: 准确率
    """
    try:
        feature_dim = support_features.shape[1]

        # 特征标准化
        support_features = F.normalize(support_features, p=2, dim=1)
        query_features = F.normalize(query_features, p=2, dim=1)

        # 初始化线性探针
        linear_probe = LinearProbe(feature_dim, n_way).to(device)
        optimizer = optim.Adam(linear_probe.parameters(), lr=0.01, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # 简化训练：只训练30轮
        linear_probe.train()
        for epoch in range(30):
            optimizer.zero_grad()
            logits = linear_probe(support_features)
            loss = criterion(logits, support_labels)
            loss.backward()
            optimizer.step()

            # 早停：训练精度达到95%就停止
            if epoch % 10 == 0:
                with torch.no_grad():
                    train_preds = torch.argmax(logits, dim=1)
                    train_acc = (train_preds == support_labels).float().mean().item()
                    if train_acc >= 0.95:
                        break

        # 测试
        linear_probe.eval()
        with torch.no_grad():
            query_logits = linear_probe(query_features)
            predictions = torch.argmax(query_logits, dim=1)
            correct = (predictions == query_labels).float().sum().item()
            accuracy = correct / len(query_labels) if len(query_labels) > 0 else 0.0

        return accuracy

    except Exception as e:
        tqdm.write(f"Linear probe evaluation error: {e}")
        return 0.0


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
