"""
Score训练器 - 简化版本
支持双曲分数网络训练，集成采样质量监控
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import trange, tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_utils import load_config, save_config
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch, load_model
from utils.manifolds_utils import get_manifold
from utils.protos_utils import compute_protos_from
from utils.sampler import Sampler
from models.GraphVAE import GraphVAE
import ml_collections


class ScoreTrainer:
    """Score训练器"""

    def __init__(self, config_path, vae_checkpoint_path):
        # 加载配置
        self.config = load_config(config_path)
        self.vae_checkpoint_path = vae_checkpoint_path

        # 设置基本参数
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.run_name = self.config.run_name

        # 初始化wandb
        self._init_wandb()

        # 加载数据集
        self.dataset = MyDataset(self.config.data, self.config.fsl_task)
        self.train_loader, self.test_loader = self.dataset.get_loaders()

        # 加载预训练编码器
        self._load_encoder()

        # 初始化Score模型
        self._init_score_models()

        # 创建保存目录
        self.save_dir = os.path.join(
            self.config.paths.save_dir, f"{self.config.exp_name}_score", self.config.timestamp
        )
        os.makedirs(self.save_dir, exist_ok=True)

        tqdm.write(f"Score Trainer initialized: {self.run_name}")
        tqdm.write(f"Save directory: {self.save_dir}")
        tqdm.write(f"Device: {self.device}")

    def _init_wandb(self):
        """初始化wandb"""
        mode = (
            "disabled"
            if self.config.debug
            else ("online" if self.config.wandb.online else "offline")
        )

        wandb.init(
            project=f"{self.config.wandb.project}_Score",
            entity=self.config.wandb.entity,
            name=f"{self.run_name}_score",
            config=self.config.to_dict(),
            mode=mode,
            dir=os.path.join("logs", "wandb"),
        )

    def _load_encoder(self):
        """加载预训练编码器"""
        tqdm.write(f"Loading VAE encoder from: {self.vae_checkpoint_path}")
        checkpoint = torch.load(
            self.vae_checkpoint_path, map_location=self.device, weights_only=False
        )

        # 重建VAE模型
        vae_config = checkpoint["model_config"]

        # 构造GraphVAE期望的配置格式
        from types import SimpleNamespace

        model_config = SimpleNamespace()
        model_config.encoder_config = vae_config["vae"]["encoder"]
        model_config.decoder_config = vae_config["vae"]["decoder"]
        model_config.pred_node_class = vae_config["vae"]["loss"]["pred_node_class"]
        model_config.pred_edge = vae_config["vae"]["loss"]["pred_edge"]
        model_config.pred_graph_class = vae_config["vae"]["loss"]["pred_graph_class"]
        model_config.use_kl_loss = vae_config["vae"]["loss"]["use_kl_loss"]
        model_config.use_base_proto_loss = vae_config["vae"]["loss"]["use_base_proto_loss"]
        model_config.use_sep_proto_loss = vae_config["vae"]["loss"]["use_sep_proto_loss"]
        model_config.latent_dim = vae_config["vae"]["encoder"]["latent_feature_dim"]
        model_config.device = self.device

        self.vae_model = GraphVAE(model_config)
        self.vae_model.load_state_dict(checkpoint["model_state_dict"])
        self.vae_model.to(self.device)
        self.vae_model.eval()

        # 提取编码器
        self.encoder = self.vae_model.encoder
        self.encoder.requires_grad_(False)
        self.manifold = self.encoder.manifold

        tqdm.write(f"✓ Encoder loaded with manifold: {self.manifold.__class__.__name__}")

    def _init_score_models(self):
        """初始化Score模型"""
        # 准备X网络配置
        x_config = dict(self.config.score.x.to_dict())
        x_config.update(
            {
                "max_feat_num": self.config.data.max_feat_num,
                "latent_feature_dim": self.config.data.max_feat_num,
                "manifold": self.manifold,
            }
        )

        # 准备Adj网络配置
        adj_config = dict(self.config.score.adj.to_dict())
        adj_config.update(
            {
                "max_feat_num": self.config.data.max_feat_num,
                "max_node_num": self.config.data.max_node_num,
                "latent_feature_dim": self.config.data.max_feat_num,
                "manifold": self.manifold,
            }
        )

        # 创建模型
        self.model_x = load_model(x_config, self.device)
        self.model_adj = load_model(adj_config, self.device)

        # 创建优化器
        self.optimizer_x = optim.Adam(
            self.model_x.parameters(),
            lr=self.config.score.train.lr,
            weight_decay=self.config.score.train.weight_decay,
        )
        self.optimizer_adj = optim.Adam(
            self.model_adj.parameters(),
            lr=self.config.score.train.lr,
            weight_decay=self.config.score.train.weight_decay,
        )

        # 创建学习率调度器
        if self.config.score.train.lr_schedule:
            self.scheduler_x = optim.lr_scheduler.ExponentialLR(
                self.optimizer_x, gamma=self.config.score.train.lr_decay
            )
            self.scheduler_adj = optim.lr_scheduler.ExponentialLR(
                self.optimizer_adj, gamma=self.config.score.train.lr_decay
            )
        else:
            self.scheduler_x = None
            self.scheduler_adj = None

        # 计算原型
        self.protos_train = compute_protos_from(self.encoder, self.train_loader, self.device)
        self.protos_test = compute_protos_from(self.encoder, self.test_loader, self.device)

        tqdm.write(f"✓ Score models initialized")
        tqdm.write(f"  X model parameters: {sum(p.numel() for p in self.model_x.parameters()):,}")
        tqdm.write(
            f"  Adj model parameters: {sum(p.numel() for p in self.model_adj.parameters()):,}"
        )

    def _sample_evaluation(self, epoch):
        """采样质量评估"""
        # 创建采样器配置
        sampler_config = self.config.sampler.to_dict()
        sampler_config.update(
            {
                "ckp_path": os.path.join(self.save_dir, "current.pth"),
                "k_augment": 10,  # 采样数量
            }
        )

        # 保存当前模型用于采样
        self._save_current_checkpoint(epoch)

        # 创建采样器
        temp_config = self.config.to_dict()
        temp_config["sampler"] = sampler_config
        temp_config = load_config(temp_config)

        try:
            sampler = Sampler(temp_config)
            # 执行采样
            sample_results = sampler.sample(independent=False)
            return sample_results.get("validity", 0.0)
        except Exception as e:
            tqdm.write(f"Error in sampling evaluation: {e}")
            return 0.0

    def train(self):
        """主训练循环"""
        tqdm.write(f"Starting Score training: {self.run_name}")

        best_test_loss = float("inf")
        best_sample_quality = 0.0

        progress_bar = tqdm(
            range(self.config.score.train.num_epochs),
            desc="Training",
            ncols=100,
            leave=True,
            ascii=True,
        )

        for epoch in progress_bar:
            # 训练阶段 - 每个epoch后都提交训练loss
            train_losses = self._train_epoch()
            mean_train_x = np.mean(train_losses["x"])
            mean_train_adj = np.mean(train_losses["adj"])
            mean_train_total = mean_train_x + mean_train_adj

            # 提交训练损失到wandb
            train_log = {
                "epoch": epoch,
                "train_x_loss": mean_train_x,
                "train_adj_loss": mean_train_adj,
                "train_total_loss": mean_train_total,
                "lr_x": self.optimizer_x.param_groups[0]["lr"],
                "lr_adj": self.optimizer_adj.param_groups[0]["lr"],
            }
            wandb.log(train_log)

            # 更新学习率
            if self.scheduler_x:
                self.scheduler_x.step()
            if self.scheduler_adj:
                self.scheduler_adj.step()

            # 检查是否需要进行测试
            should_test = (epoch % self.config.score.train.test_interval == 0) or (
                epoch == self.config.score.train.num_epochs - 1
            )

            if should_test:
                # 测试阶段
                test_losses = self._test_epoch()
                mean_test_x = np.mean(test_losses["x"])
                mean_test_adj = np.mean(test_losses["adj"])
                total_test_loss = mean_test_x + mean_test_adj

                # 采样质量评估
                sample_quality = self._sample_evaluation(epoch)

                # 提交测试损失和指标到wandb
                test_log = {
                    "epoch": epoch,
                    "test_x_loss": mean_test_x,
                    "test_adj_loss": mean_test_adj,
                    "test_total_loss": total_test_loss,
                    "sample_validity": sample_quality,
                }
                wandb.log(test_log)

                # 检查是否需要保存最佳模型
                is_best_loss = total_test_loss < best_test_loss
                is_best_sample = sample_quality > best_sample_quality

                if is_best_loss:
                    best_test_loss = total_test_loss
                    self._save_checkpoint(epoch, total_test_loss, sample_quality, "best_loss")
                    tqdm.write(f"✓ New best loss: {total_test_loss:.6f}")

                if is_best_sample:
                    best_sample_quality = sample_quality
                    self._save_checkpoint(epoch, total_test_loss, sample_quality, "best_sample")
                    tqdm.write(f"✓ New best sample quality: {sample_quality:.4f}")

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "Train": f"{mean_train_total:.6f}",
                        "Test": f"{total_test_loss:.6f}",
                        "Sample": f"{sample_quality:.4f}",
                        "Best": f"{min(best_test_loss, total_test_loss):.6f}",
                    }
                )

                tqdm.write(
                    f"Epoch {epoch}: Train X={mean_train_x:.6f}, Adj={mean_train_adj:.6f} | "
                    f"Test X={mean_test_x:.6f}, Adj={mean_test_adj:.6f} | Sample={sample_quality:.4f}"
                )
            else:
                # 只更新进度条显示训练loss
                progress_bar.set_postfix(
                    {
                        "Train": f"{mean_train_total:.6f}",
                        "Test": "N/A",
                        "Sample": "N/A",
                        "Best": f"{best_test_loss:.6f}",
                    }
                )

        # 保存最终模型
        final_test_losses = self._test_epoch()
        final_mean_test_x = np.mean(final_test_losses["x"])
        final_mean_test_adj = np.mean(final_test_losses["adj"])
        final_total_test_loss = final_mean_test_x + final_mean_test_adj
        final_sample_quality = self._sample_evaluation(self.config.score.train.num_epochs - 1)

        self._save_checkpoint(
            self.config.score.train.num_epochs - 1,
            final_total_test_loss,
            final_sample_quality,
            "final",
        )

        tqdm.write(
            f"Training completed. Best test loss: {best_test_loss:.6f}, Best sample quality: {best_sample_quality:.4f}"
        )
        return self.save_dir

    def _train_epoch(self):
        """训练一个epoch"""
        self.model_x.train()
        self.model_adj.train()
        losses = {"x": [], "adj": []}

        for batch in self.train_loader:
            x, adj, labels = load_batch(batch, self.device)

            # 计算损失（这里需要实现具体的score matching损失）
            loss_x = self._compute_score_loss_x(x, adj, labels)
            loss_adj = self._compute_score_loss_adj(x, adj, labels)

            # X网络更新
            self.optimizer_x.zero_grad()
            loss_x.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_x.parameters(), self.config.score.train.grad_norm
            )
            self.optimizer_x.step()

            # Adj网络更新
            self.optimizer_adj.zero_grad()
            loss_adj.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_adj.parameters(), self.config.score.train.grad_norm
            )
            self.optimizer_adj.step()

            # 记录损失
            losses["x"].append(loss_x.item())
            losses["adj"].append(loss_adj.item())

        return losses

    def _test_epoch(self):
        """测试一个epoch"""
        self.model_x.eval()
        self.model_adj.eval()
        losses = {"x": [], "adj": []}

        with torch.no_grad():
            for batch in self.test_loader:
                x, adj, labels = load_batch(batch, self.device)

                # 计算损失
                loss_x = self._compute_score_loss_x(x, adj, labels)
                loss_adj = self._compute_score_loss_adj(x, adj, labels)

                # 记录损失
                losses["x"].append(loss_x.item())
                losses["adj"].append(loss_adj.item())

        return losses

    def _compute_score_loss_x(self, x, adj, labels):
        """计算X网络的score matching损失"""
        # 这里是简化版本，实际需要实现完整的score matching损失
        # 添加噪声
        noise = torch.randn_like(x) * 0.1
        x_noisy = x + noise

        # 计算分数
        score = self.model_x(x_noisy, adj)

        # 简化的损失计算
        loss = torch.mean((score + noise / 0.01) ** 2)
        return loss

    def _compute_score_loss_adj(self, x, adj, labels):
        """计算Adj网络的score matching损失"""
        # 这里是简化版本，实际需要实现完整的score matching损失
        # 添加噪声
        noise = torch.randn_like(adj) * 0.1
        adj_noisy = adj + noise

        # 计算分数
        score = self.model_adj(x, adj_noisy)

        # 简化的损失计算
        loss = torch.mean((score + noise / 0.01) ** 2)
        return loss

    def _save_current_checkpoint(self, epoch):
        """保存当前检查点用于采样"""
        checkpoint = {
            "epoch": epoch,
            "model_config": self.config.to_dict(),
            "params_x": self.config.score.x.to_dict(),
            "params_adj": self.config.score.adj.to_dict(),
            "x_state_dict": self.model_x.state_dict(),
            "adj_state_dict": self.model_adj.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.save_dir, "current.pth"))

    def _save_checkpoint(self, epoch, test_loss, sample_quality, checkpoint_type):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_config": self.config.to_dict(),
            "params_x": self.config.score.x.to_dict(),
            "params_adj": self.config.score.adj.to_dict(),
            "x_state_dict": self.model_x.state_dict(),
            "adj_state_dict": self.model_adj.state_dict(),
            "optimizer_x_state_dict": self.optimizer_x.state_dict(),
            "optimizer_adj_state_dict": self.optimizer_adj.state_dict(),
            "test_loss": test_loss,
            "sample_quality": sample_quality,
        }

        if self.scheduler_x:
            checkpoint["scheduler_x_state_dict"] = self.scheduler_x.state_dict()
        if self.scheduler_adj:
            checkpoint["scheduler_adj_state_dict"] = self.scheduler_adj.state_dict()

        # 保存指定类型的检查点
        checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_type}.pth")
        torch.save(checkpoint, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Score Trainer")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VAE检查点路径")
    args = parser.parse_args()

    # 创建训练器并开始训练
    trainer = ScoreTrainer(args.config, args.vae_checkpoint)
    save_dir = trainer.train()

    tqdm.write(f"Score training completed. Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
