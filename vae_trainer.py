"""
VAE训练器 - 简化版本
支持双曲图自编码器训练，集成元测试监控
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import trange, tqdm

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


class VAETrainer:
    """VAE训练器"""

    def __init__(self, config_path):
        # 加载配置
        self.config = load_config(config_path)

        # 设置基本参数
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.run_name = self.config.run_name

        # 初始化wandb
        self._init_wandb()

        # 加载数据集
        self.dataset = MyDataset(self.config.data, self.config.fsl_task)
        self.train_loader, self.test_loader = self.dataset.get_loaders()

        # 初始化模型
        self._init_model()

        # 创建保存目录
        self.save_dir = os.path.join(
            self.config.paths.save_dir, self.config.exp_name, self.config.timestamp
        )
        os.makedirs(self.save_dir, exist_ok=True)

        tqdm.write(f"VAE Trainer initialized: {self.run_name}")
        tqdm.write(f"Save directory: {self.save_dir}")
        tqdm.write(f"Device: {self.device}")

        # Meta-test设置
        self.meta_test_enabled = (
            hasattr(self.config, "fsl_task") and self.config.fsl_task is not None
        )
        if self.meta_test_enabled:
            tqdm.write(
                f"✓ Meta-test enabled with {self.config.fsl_task.N_way}-way {self.config.fsl_task.K_shot}-shot"
            )
        else:
            tqdm.write("✗ Meta-test disabled: no fsl_task config")

        # 早停机制初始化
        self.enable_early_stopping = getattr(self.config.vae.train, "enable_early_stopping", True)

        if self.enable_early_stopping:
            early_stop_patience = getattr(self.config.vae.train, "early_stop_patience", 5)
            early_stop_min_delta = getattr(self.config.vae.train, "early_stop_min_delta", 0.01)
            self.early_stopping = EarlyStopping(
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
                mode="max",  # Meta-Test准确率越高越好
            )
            tqdm.write(
                f"✓ Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}"
            )
        else:
            self.early_stopping = None
            tqdm.write("✗ Early stopping disabled - full training curve will be recorded")

    def _init_wandb(self):
        """初始化wandb"""
        mode = (
            "disabled"
            if self.config.debug
            else ("online" if self.config.wandb.online else "offline")
        )

        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.run_name,
            config=self.config.to_dict(),
            mode=mode,
            dir=os.path.join("logs", "wandb"),
        )

    def _init_model(self):
        """初始化VAE模型"""
        # 创建GraphVAE配置对象
        from types import SimpleNamespace

        vae_config = SimpleNamespace()
        vae_config.pred_node_class = self.config.vae.loss.pred_node_class
        vae_config.pred_edge = self.config.vae.loss.pred_edge
        vae_config.pred_graph_class = self.config.vae.loss.pred_graph_class
        vae_config.use_kl_loss = self.config.vae.loss.use_kl_loss
        vae_config.use_base_proto_loss = self.config.vae.loss.use_base_proto_loss
        vae_config.use_sep_proto_loss = self.config.vae.loss.use_sep_proto_loss

        # 设置编码器和解码器配置，补充缺失的字段
        vae_config.encoder_config = self.config.vae.encoder
        vae_config.encoder_config.input_feature_dim = self.config.data.max_feat_num

        vae_config.decoder_config = self.config.vae.decoder
        vae_config.decoder_config.latent_feature_dim = self.config.vae.encoder.latent_feature_dim
        vae_config.decoder_config.output_feature_dim = self.config.data.max_feat_num

        vae_config.latent_dim = self.config.vae.encoder.latent_feature_dim
        vae_config.device = self.device

        # 创建GraphVAE
        self.model = GraphVAE(vae_config).to(self.device)

        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.vae.train.lr,
            weight_decay=self.config.vae.train.weight_decay,
        )

        # 创建学习率调度器 - 改进版
        if self.config.vae.train.lr_schedule:
            # 支持多种调度策略
            scheduler_type = getattr(self.config.vae.train, "scheduler_type", "exponential")

            if scheduler_type == "exponential":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.config.vae.train.lr_decay
                )
            elif scheduler_type == "cosine":
                # Cosine Annealing调度
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.vae.train.num_epochs,
                    eta_min=self.config.vae.train.lr * 0.01,  # 最低学习率为初始的1%
                )
            elif scheduler_type == "step":
                # 阶梯式衰减
                step_size = getattr(self.config.vae.train, "lr_step_size", 100)
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=self.config.vae.train.lr_decay
                )
            else:
                # 默认使用指数衰减
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.config.vae.train.lr_decay
                )

            # Warm-up调度器支持
            warmup_epochs = getattr(self.config.vae.train, "warmup_epochs", 0)
            if warmup_epochs > 0:
                from torch.optim.lr_scheduler import LambdaLR

                def warmup_lambda(epoch):
                    if epoch < warmup_epochs:
                        return (epoch + 1) / warmup_epochs
                    return 1.0

                self.warmup_scheduler = LambdaLR(self.optimizer, warmup_lambda)
                self.use_warmup = True
                tqdm.write(f"✓ Warm-up enabled: {warmup_epochs} epochs")
            else:
                self.use_warmup = False

            tqdm.write(f"✓ LR Scheduler: {scheduler_type}, decay: {self.config.vae.train.lr_decay}")
        else:
            self.scheduler = None
            self.use_warmup = False

        # 获取编码器
        self.encoder = self.model.encoder

        tqdm.write(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 初始化元测试相关组件
        self._init_meta_test_components()

    def _init_meta_test_components(self):
        """初始化meta-test评估组件 - 简化版本"""
        if not hasattr(self.config, "fsl_task") or self.config.fsl_task is None:
            print("No FSL task config found. Meta-test evaluation disabled.")
            self.meta_test_enabled = False
            return

        self.meta_test_enabled = True
        fsl_config = self.config.fsl_task

        # 从配置读取参数
        N_way = fsl_config.N_way
        meta_test_tasks = getattr(fsl_config, "meta_test_tasks", 10)

        # 简单线性探针 - 直接使用固定维度
        latent_dim = self.config.vae.encoder.latent_feature_dim
        self.linear_probe = torch.nn.Linear(latent_dim, N_way).to(self.device)

        print(f"✓ Meta-test components initialized:")
        print(f"  N-way: {N_way}, K-shot: {fsl_config.K_shot}, R-query: {fsl_config.R_query}")
        print(f"  Meta-test tasks: {meta_test_tasks}")
        print(f"  Linear probe: {latent_dim} -> {N_way}")

    def train(self):
        """主训练循环"""
        tqdm.write(f"Starting VAE training: {self.run_name}")

        best_test_loss = float("inf")
        best_meta_test_acc = 0.0

        progress_bar = tqdm(
            range(self.config.vae.train.num_epochs),
            desc="Training",
            ncols=100,
            leave=True,
            ascii=True,
        )

        for epoch in progress_bar:
            # 训练阶段 - 每个epoch后都提交训练loss
            train_losses = self._train_epoch()
            mean_train_loss = np.mean(train_losses["total"])

            # 提交训练损失到wandb
            train_log = {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "train_rec_loss": np.mean(train_losses["rec"]),
                "train_kl_loss": np.mean(train_losses["kl"]),
                "train_edge_loss": np.mean(train_losses["edge"]),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            wandb.log(train_log)

            # 更新学习率
            if self.scheduler:
                if self.use_warmup and epoch < getattr(self.config.vae.train, "warmup_epochs", 0):
                    # Warm-up阶段
                    self.warmup_scheduler.step()
                else:
                    # 正常调度阶段
                    self.scheduler.step()

            # 检查是否需要进行测试
            should_test = (epoch % self.config.vae.train.test_interval == 0) or (
                epoch == self.config.vae.train.num_epochs - 1
            )

            if should_test:
                # 测试阶段
                test_losses = self._test_epoch()
                mean_test_loss = np.mean(test_losses["total"])

                # 元测试评估
                meta_test_acc = 0.0
                if self.meta_test_enabled:
                    meta_test_acc = self._meta_test_evaluation(epoch)

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
                    self._save_checkpoint(epoch, mean_test_loss, meta_test_acc, "best_loss")
                    progress_bar.write(f"✓ New best loss: {mean_test_loss:.6f}")

                if is_best_meta_acc:
                    best_meta_test_acc = meta_test_acc
                    self._save_checkpoint(epoch, mean_test_loss, meta_test_acc, "best_meta_acc")
                    progress_bar.write(f"✓ New best meta-acc: {meta_test_acc:.4f}")

                # 早停检查 - 只在启用时执行
                if (
                    self.enable_early_stopping
                    and self.meta_test_enabled
                    and self.early_stopping is not None
                ):
                    should_early_stop = self.early_stopping(meta_test_acc)
                    if should_early_stop:
                        progress_bar.write(f"🛑 Early stopping triggered at epoch {epoch}")
                        progress_bar.write(
                            f"   Best Meta-Test Acc: {self.early_stopping.best_score:.4f}"
                        )
                        progress_bar.write(
                            f"   No improvement for {self.early_stopping.patience} consecutive evaluations"
                        )
                        # 保存早停时的模型
                        self._save_checkpoint(epoch, mean_test_loss, meta_test_acc, "early_stop")
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
                # 非测试epoch - 只显示训练loss和最佳meta准确率
                progress_bar.set_postfix(
                    {
                        "Train": f"{mean_train_loss:.6f}",
                        "Best-Meta": f"{best_meta_test_acc:.4f}",
                    }
                )

        # 保存最终模型
        final_test_losses = self._test_epoch()
        final_mean_test_loss = np.mean(final_test_losses["total"])
        final_meta_test_acc = 0.0
        if self.meta_test_enabled:
            final_meta_test_acc = self._meta_test_evaluation(self.config.vae.train.num_epochs - 1)

        self._save_checkpoint(
            self.config.vae.train.num_epochs - 1, final_mean_test_loss, final_meta_test_acc, "final"
        )

        tqdm.write(
            f"Training completed. Best test loss: {best_test_loss:.6f}, Best meta-test acc: {best_meta_test_acc:.4f}"
        )
        return self.save_dir

    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        losses = {"total": [], "rec": [], "kl": [], "edge": []}

        for batch in self.train_loader:
            x, adj, labels = load_batch(batch, self.device)

            self.optimizer.zero_grad()

            # 前向传播
            (
                rec_loss,
                kl_loss,
                edge_loss,
                base_proto_loss,
                sep_proto_loss,
                graph_classification_loss,
                acc_proto,
            ) = self.model(x, adj, labels)

            # 计算总损失
            total_loss = (
                self.config.vae.train.rec_weight * rec_loss
                + self.config.vae.train.kl_regularization * kl_loss
                + self.config.vae.train.edge_weight * edge_loss
                + self.config.vae.train.base_proto_weight * base_proto_loss
                + self.config.vae.train.sep_proto_weight * sep_proto_loss
                + self.config.vae.train.graph_classification_weight * graph_classification_loss
            )

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.vae.train.grad_norm)
            self.optimizer.step()

            # 记录损失
            losses["total"].append(total_loss.item())
            losses["rec"].append(rec_loss.item())
            losses["kl"].append(kl_loss.item())
            losses["edge"].append(edge_loss.item())

        return losses

    def _test_epoch(self):
        """测试一个epoch"""
        self.model.eval()
        losses = {"total": [], "rec": [], "kl": [], "edge": []}

        with torch.no_grad():
            for batch in self.test_loader:
                x, adj, labels = load_batch(batch, self.device)

                # 前向传播
                (
                    rec_loss,
                    kl_loss,
                    edge_loss,
                    base_proto_loss,
                    sep_proto_loss,
                    graph_classification_loss,
                    acc_proto,
                ) = self.model(x, adj, labels)

                # 计算总损失
                total_loss = (
                    self.config.vae.train.rec_weight * rec_loss
                    + self.config.vae.train.kl_regularization * kl_loss
                    + self.config.vae.train.edge_weight * edge_loss
                    + self.config.vae.train.base_proto_weight * base_proto_loss
                    + self.config.vae.train.sep_proto_weight * sep_proto_loss
                    + self.config.vae.train.graph_classification_weight * graph_classification_loss
                )

                # 记录损失
                losses["total"].append(total_loss.item())
                losses["rec"].append(rec_loss.item())
                losses["kl"].append(kl_loss.item())
                losses["edge"].append(edge_loss.item())

        return losses

    def _meta_test_evaluation(self, epoch):
        """Meta-test evaluation using linear probing"""
        # 使用配置文件中的FSL参数
        N_way = self.config.fsl_task.N_way
        K_shot = self.config.fsl_task.K_shot
        R_query = self.config.fsl_task.R_query
        meta_test_tasks = getattr(self.config.fsl_task, "meta_test_tasks", 10)

        self.encoder.eval()
        all_task_accuracies = []

        try:
            for task_idx in range(meta_test_tasks):
                task = self.dataset.sample_one_task(
                    is_train=False,
                    N_way=N_way,
                    K_shot=K_shot,
                    R_query=R_query,
                    query_pool_start_index=task_idx * R_query,
                )

                if task is None:
                    continue

                # 将任务数据移到设备
                support_x = task["support_set"]["x"].to(self.device)
                support_adj = task["support_set"]["adj"].to(self.device)
                support_labels = task["support_set"]["label"].to(self.device)
                query_x = task["query_set"]["x"].to(self.device)
                query_adj = task["query_set"]["adj"].to(self.device)
                query_labels = task["query_set"]["label"].to(self.device)

                # 提取特征
                support_features = self._extract_features(support_x, support_adj)
                query_features = self._extract_features(query_x, query_adj)

                # 为每个任务创建新的线性探针
                actual_N_way = task.get("N_way", N_way)
                latent_dim = self.config.vae.encoder.latent_feature_dim
                task_linear_probe = torch.nn.Linear(latent_dim, actual_N_way).to(self.device)

                # 训练并测试线性探针
                accuracy = self._train_and_test_probe(
                    support_features,
                    support_labels,
                    query_features,
                    query_labels,
                    task_linear_probe,
                )
                all_task_accuracies.append(accuracy)

        except Exception as e:
            tqdm.write(f"Error in meta-test evaluation: {e}")
            return 0.0

        # 计算最终结果
        if all_task_accuracies:
            mean_accuracy = np.mean(all_task_accuracies)
            return mean_accuracy
        else:
            return 0.0

    def _extract_features(self, x_batch, adj_batch):
        """使用编码器提取特征"""
        with torch.no_grad():
            # 生成node_mask
            from utils.graph_utils import node_flags

            node_mask = node_flags(adj_batch)
            node_mask = node_mask.unsqueeze(-1)  # 增加最后一个维度

            # 使用编码器提取特征
            posterior = self.encoder(x_batch, adj_batch, node_mask)
            z_mu = posterior.mode()  # 获取后验分布的模式

            # 对于图级别的分类，我们需要聚合节点特征
            # 使用平均池化，同时考虑node_mask
            node_mask_for_pooling = node_mask.squeeze(-1)  # [batch_size, num_nodes]
            masked_features = z_mu * node_mask.expand_as(z_mu)  # 应用mask

            # 计算每个图的有效节点数
            num_valid_nodes = node_mask_for_pooling.sum(dim=1, keepdim=True)  # [batch_size, 1]
            num_valid_nodes = torch.clamp(num_valid_nodes, min=1.0)  # 避免除零

            # 平均池化得到图级特征
            graph_features = (
                masked_features.sum(dim=1) / num_valid_nodes
            )  # [batch_size, latent_dim]

            return graph_features

    def _train_and_test_probe(
        self, support_features, support_labels, query_features, query_labels, task_linear_probe
    ):
        """训练线性探针并测试 - 简化版本"""
        # 重置线性探针参数
        torch.nn.init.xavier_uniform_(task_linear_probe.weight)
        torch.nn.init.zeros_(task_linear_probe.bias)

        # 创建优化器
        optimizer = torch.optim.Adam(task_linear_probe.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # 早停参数
        best_loss = float("inf")
        patience = 10
        patience_counter = 0
        max_epochs = 100

        # 训练线性探针
        task_linear_probe.train()
        for epoch in range(max_epochs):
            optimizer.zero_grad()

            # 前向传播
            logits = task_linear_probe(support_features)
            loss = criterion(logits, support_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 早停检查
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # 测试
        task_linear_probe.eval()
        with torch.no_grad():
            query_logits = task_linear_probe(query_features)
            predictions = torch.argmax(query_logits, dim=1)
            correct = (predictions == query_labels).float().sum().item()
            accuracy = correct / len(query_labels)

        return accuracy

    def _save_checkpoint(self, epoch, test_loss, meta_test_acc, checkpoint_type):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_config": self.config.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "test_loss": test_loss,
            "meta_test_acc": meta_test_acc,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # 保存指定类型的检查点
        checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_type}.pth")
        torch.save(checkpoint, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="VAE Trainer")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    # 创建训练器并开始训练
    trainer = VAETrainer(args.config)
    save_dir = trainer.train()

    tqdm.write(f"VAE training completed. Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
