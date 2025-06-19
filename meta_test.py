"""
Meta-test训练器 - 简化版本
支持可选的任务扩充+分类头微调+元学习评估
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
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_utils import load_config, save_config
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch
from utils.graph_utils import node_flags
from models.GraphVAE import GraphVAE
from models.Decoders import Classifier

# 任务扩充相关导入
from utils.task_sampler import TaskSamplerConfig, create_meta_task_sampler


class MetaTestTrainer:
    """Meta-test训练器 - 支持可选的任务扩充"""

    def __init__(self, config_path, vae_checkpoint_path, score_checkpoint_path=None):
        # 加载配置
        self.config = load_config(config_path)
        self.vae_checkpoint_path = vae_checkpoint_path
        self.score_checkpoint_path = score_checkpoint_path

        # 设置基本参数
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config.device)
        self.run_name = self.config.run_name

        # 初始化wandb
        self._init_wandb()

        # 加载数据集
        self.dataset = MyDataset(self.config.data, self.config.fsl_task)
        self.train_loader, self.test_loader = self.dataset.get_loaders()

        # 加载预训练编码器
        self._load_encoder()

        # 可选：初始化任务扩充器
        self.task_sampler = None
        if self._should_use_task_augmentation():
            self._init_task_sampler()

        # 创建保存目录
        self.save_dir = os.path.join(
            self.config.paths.save_dir, f"{self.config.exp_name}_meta", self.config.timestamp
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # 打印初始化信息
        self._print_initialization_info()

    def _should_use_task_augmentation(self):
        """检查是否应该使用任务扩充"""
        return (
            hasattr(self.config, "meta_test")
            and hasattr(self.config.meta_test, "task_augmentation")
            and getattr(self.config.meta_test.task_augmentation, "enabled", False)
            and self.score_checkpoint_path is not None
        )

    def _init_task_sampler(self):
        """初始化任务扩充器"""
        try:
            tqdm.write("🔧 Initializing task augmentation sampler...")

            # 从完整配置中提取任务采样器配置
            task_sampler_config = TaskSamplerConfig.from_full_config(self.config)

            # 获取编码器检查点路径
            encoder_ckpt_path = getattr(
                self.config.meta_test.task_augmentation,
                "encoder_checkpoint_path",
                self.vae_checkpoint_path,  # 默认使用VAE检查点
            )

            # 创建任务采样器
            self.task_sampler = create_meta_task_sampler(
                config=task_sampler_config,
                score_ckpt_path=self.score_checkpoint_path,
                encoder_ckpt_path=encoder_ckpt_path,
            )

            tqdm.write("✓ Task augmentation sampler initialized successfully")

        except Exception as e:
            tqdm.write(f"⚠️ Failed to initialize task sampler: {e}")
            tqdm.write("   Continuing without task augmentation...")
            self.task_sampler = None

    def _print_initialization_info(self):
        """打印初始化信息"""
        tqdm.write(f"Meta-test Trainer initialized: {self.run_name}")
        tqdm.write(f"Save directory: {self.save_dir}")
        tqdm.write(f"Device: {self.device}")
        tqdm.write(f"Task augmentation: {'enabled' if self.task_sampler else 'disabled'}")

        if self.task_sampler:
            aug_config = self.config.meta_test.task_augmentation
            tqdm.write(f"  - k_augment: {getattr(aug_config, 'k_augment', 2)}")
            tqdm.write(f"  - finetune_steps: {getattr(aug_config, 'finetune_steps', 10)}")

    def _init_wandb(self):
        """初始化wandb"""
        mode = (
            "disabled"
            if self.config.debug
            else ("online" if self.config.wandb.online else "offline")
        )

        wandb.init(
            project=f"{self.config.wandb.project}_Meta",
            entity=self.config.wandb.entity,
            name=f"{self.run_name}_meta",
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
        self.vae_model = GraphVAE(
            pred_node_class=vae_config["vae"]["loss"]["pred_node_class"],
            pred_edge=vae_config["vae"]["loss"]["pred_edge"],
            pred_graph_class=vae_config["vae"]["loss"]["pred_graph_class"],
            use_kl_loss=vae_config["vae"]["loss"]["use_kl_loss"],
            use_base_proto_loss=vae_config["vae"]["loss"]["use_base_proto_loss"],
            use_sep_proto_loss=vae_config["vae"]["loss"]["use_sep_proto_loss"],
            encoder_config=vae_config["vae"]["encoder"],
            decoder_config=vae_config["vae"]["decoder"],
            latent_dim=vae_config["vae"]["encoder"]["latent_feature_dim"],
            device=self.device,
        )
        self.vae_model.load_state_dict(checkpoint["model_state_dict"])
        self.vae_model.to(self.device)
        self.vae_model.eval()

        # 提取编码器
        self.encoder = self.vae_model.encoder
        self.encoder.requires_grad_(False)

        tqdm.write(f"✓ Encoder loaded")

    def _get_embeddings(self, x, adj):
        """获取图嵌入"""
        mask = node_flags(adj).unsqueeze(-1)

        with torch.no_grad():
            self.encoder.eval()
            z = self.encoder(x, adj, mask)

            # 处理分布输出
            if hasattr(z, "mode"):
                z = z.mode()

            # 转换到欧几里得空间
            if hasattr(self.encoder, "manifold") and self.encoder.manifold:
                z = self.encoder.manifold.logmap0(z)

            # 池化
            pooling_method = self.config.meta_test.embedding.pooling_method
            if pooling_method == "mean":
                embeddings = z.mean(dim=1)
            elif pooling_method == "max":
                embeddings = z.max(dim=1).values
            elif pooling_method == "mean_max":
                mean_emb = z.mean(dim=1)
                max_emb = z.max(dim=1).values
                embeddings = torch.cat([mean_emb, max_emb], dim=-1)
            else:
                embeddings = z.mean(dim=1)  # 默认使用mean

            # 可选的标准化
            if self.config.meta_test.embedding.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _augment_task_if_enabled(self, task):
        """如果启用，对任务进行扩充"""
        if self.task_sampler is None:
            return task

        try:
            # 获取扩充参数
            aug_config = self.config.meta_test.task_augmentation
            k_augment = getattr(aug_config, "k_augment", 2)
            finetune_steps = getattr(aug_config, "finetune_steps", 10)
            learning_rate = getattr(aug_config, "learning_rate", 1e-3)

            # 执行任务扩充
            augmented_task = self.task_sampler.augment_task(
                task=task,
                k_augment=k_augment,
                finetune_steps=finetune_steps,
                learning_rate=learning_rate,
            )

            return augmented_task

        except Exception as e:
            tqdm.write(f"⚠️ Task augmentation failed: {e}")
            tqdm.write("   Using original task...")
            return task

    def _train_classifier_on_task(self, task):
        """在单个任务上训练分类头"""
        # 可选的任务扩充
        if self.task_sampler is not None:
            task = self._augment_task_if_enabled(task)

        # 获取支持集和查询集数据
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_label = task["support_set"]["label"].to(self.device)

        query_x = task["query_set"]["x"].to(self.device)
        query_adj = task["query_set"]["adj"].to(self.device)
        query_label = task["query_set"]["label"].to(self.device)

        # 获取嵌入向量
        support_emb = self._get_embeddings(support_x, support_adj)
        query_emb = self._get_embeddings(query_x, query_adj)

        # 计算类别数和嵌入维度
        n_way = len(torch.unique(support_label))
        emb_dim = support_emb.shape[-1]

        # 根据池化方法确定分类器输入维度
        pooling_method = self.config.meta_test.embedding.pooling_method
        if pooling_method == "mean_max":
            model_dim_for_classifier = emb_dim // 2
        else:
            model_dim_for_classifier = emb_dim

        # 创建分类器
        classifier = Classifier(
            model_dim=model_dim_for_classifier,
            num_classes=n_way,
            classifier_dropout=self.config.meta_test.classifier.dropout,
            classifier_bias=self.config.meta_test.classifier.bias,
            manifold=None,
        ).to(self.device)

        # 优化器和损失函数
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=self.config.meta_test.classifier.lr,
            weight_decay=self.config.meta_test.classifier.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        # 训练分类器
        best_loss = float("inf")
        patience_counter = 0

        classifier.train()
        for epoch in range(self.config.meta_test.classifier.epochs):
            optimizer.zero_grad()

            # 前向传播
            logits = classifier(support_emb)
            loss = loss_fn(logits, support_label)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 早停检查
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.meta_test.classifier.patience:
                    break

        # 在查询集上评估
        classifier.eval()
        with torch.no_grad():
            query_logits = classifier(query_emb)
            query_loss = loss_fn(query_logits, query_label)

            # 计算准确率
            _, predicted = torch.max(query_logits, 1)
            accuracy = (predicted == query_label).float().mean().item()

            # 计算其他指标
            predicted_np = predicted.cpu().numpy()
            query_label_np = query_label.cpu().numpy()

            f1 = f1_score(query_label_np, predicted_np, average="macro", zero_division=0)
            precision = precision_score(
                query_label_np, predicted_np, average="macro", zero_division=0
            )
            recall = recall_score(query_label_np, predicted_np, average="macro", zero_division=0)

        # 可选：支持集评估
        support_metrics = {}
        if self.config.meta_test.eval_support:
            with torch.no_grad():
                support_logits = classifier(support_emb)
                support_loss = loss_fn(support_logits, support_label)
                _, support_predicted = torch.max(support_logits, 1)
                support_accuracy = (support_predicted == support_label).float().mean().item()

                support_metrics = {
                    "support_loss": support_loss.item(),
                    "support_accuracy": support_accuracy,
                }

        # 记录任务扩充信息
        augmentation_metrics = {}
        if self.task_sampler is not None:
            original_support_size = task.get("original_support_size", support_x.size(0))
            current_support_size = support_x.size(0)
            augmentation_metrics = {
                "support_size_original": original_support_size,
                "support_size_augmented": current_support_size,
                "augmentation_ratio": (
                    current_support_size / original_support_size
                    if original_support_size > 0
                    else 1.0
                ),
            }

        return {
            "query_loss": query_loss.item(),
            "query_accuracy": accuracy,
            "query_f1": f1,
            "query_precision": precision,
            "query_recall": recall,
            **support_metrics,
            **augmentation_metrics,
        }

    def run_meta_test(self):
        """运行元学习测试"""
        tqdm.write(f"Starting meta-learning test: {self.run_name}")

        # 生成测试任务
        num_test_tasks = self.config.meta_test.num_test_tasks
        n_way = self.config.fsl_task.N_way
        k_shot = self.config.fsl_task.K_shot
        r_query = self.config.fsl_task.R_query

        tqdm.write(f"Running {num_test_tasks} test tasks ({n_way}-way {k_shot}-shot)")
        if self.task_sampler:
            aug_config = self.config.meta_test.task_augmentation
            k_augment = getattr(aug_config, "k_augment", 2)
            tqdm.write(f"Task augmentation enabled: k_augment={k_augment}")

        # 存储结果
        all_results = []

        # 进度条
        pbar = tqdm(range(num_test_tasks), desc="Meta-testing")

        for task_idx in pbar:
            # 生成单个任务
            task = self.dataset.sample_one_task(
                is_train=False, N_way=n_way, K_shot=k_shot, R_query=r_query
            )

            # 记录原始支持集大小（用于扩充统计）
            if self.task_sampler:
                task["original_support_size"] = task["support_set"]["x"].size(0)

            # 训练分类头并评估
            task_result = self._train_classifier_on_task(task)
            all_results.append(task_result)

            # 更新进度条
            if len(all_results) > 0:
                mean_acc = np.mean([r["query_accuracy"] for r in all_results])
                pbar.set_postfix({"mean_acc": f"{mean_acc:.4f}"})

            # 定期记录到wandb
            if (task_idx + 1) % self.config.meta_test.logging.log_interval == 0:
                self._log_intermediate_results(all_results, task_idx + 1)

        # 计算最终统计结果
        final_results = self._compute_final_statistics(all_results)

        # 记录最终结果
        self._log_final_results(final_results)

        # 保存结果
        self._save_results(final_results, all_results)

        # 打印最终结果
        self._print_final_results(final_results)

        return final_results

    def _log_intermediate_results(self, results, task_count):
        """记录中间结果到wandb"""
        if len(results) == 0:
            return

        # 计算当前的平均指标
        current_stats = {}
        for metric in ["query_accuracy", "query_f1", "query_precision", "query_recall"]:
            values = [r[metric] for r in results if metric in r]
            if values:
                current_stats[f"running_mean_{metric}"] = np.mean(values)
                current_stats[f"running_std_{metric}"] = np.std(values)

        # 任务扩充统计
        if self.task_sampler:
            aug_ratios = [r.get("augmentation_ratio", 1.0) for r in results]
            if aug_ratios:
                current_stats["running_mean_augmentation_ratio"] = np.mean(aug_ratios)

        current_stats["completed_tasks"] = task_count
        wandb.log(current_stats)

    def _compute_final_statistics(self, results):
        """计算最终统计结果"""
        if len(results) == 0:
            return {}

        final_stats = {}

        # 计算每个指标的统计值
        metrics_to_compute = [
            "query_accuracy",
            "query_f1",
            "query_precision",
            "query_recall",
            "query_loss",
            "support_accuracy",
            "support_loss",
            "augmentation_ratio",
        ]

        for metric in metrics_to_compute:
            values = [r[metric] for r in results if metric in r]
            if values:
                values = np.array(values)
                mean_val = np.mean(values)
                std_val = np.std(values)

                # 计算置信区间
                confidence_interval = std_val * 1.96 / np.sqrt(len(values))  # 95% CI

                final_stats[f"mean_{metric}"] = mean_val
                final_stats[f"std_{metric}"] = std_val
                final_stats[f"ci_{metric}"] = confidence_interval
                final_stats[f"num_samples_{metric}"] = len(values)

        final_stats["total_tasks"] = len(results)

        # 任务扩充统计
        if self.task_sampler:
            original_sizes = [r.get("support_size_original", 0) for r in results]
            augmented_sizes = [r.get("support_size_augmented", 0) for r in results]
            if original_sizes and augmented_sizes:
                final_stats["mean_original_support_size"] = np.mean(original_sizes)
                final_stats["mean_augmented_support_size"] = np.mean(augmented_sizes)

        return final_stats

    def _log_final_results(self, final_results):
        """记录最终结果到wandb"""
        wandb.log(final_results, commit=True)

        # 创建结果总结表格
        summary_data = []
        for metric in ["query_accuracy", "query_f1", "query_precision", "query_recall"]:
            if f"mean_{metric}" in final_results:
                summary_data.append(
                    [
                        metric,
                        final_results[f"mean_{metric}"],
                        final_results[f"std_{metric}"],
                        final_results[f"ci_{metric}"],
                    ]
                )

        if summary_data:
            table = wandb.Table(columns=["Metric", "Mean", "Std", "95% CI"], data=summary_data)
            wandb.log({"final_results_table": table})

    def _print_final_results(self, final_results):
        """打印最终结果"""
        tqdm.write(f"Meta-learning test completed!")
        tqdm.write(f"Final Results:")

        # 主要指标
        for metric in ["query_accuracy", "query_f1", "query_precision", "query_recall"]:
            if f"mean_{metric}" in final_results:
                mean_val = final_results[f"mean_{metric}"]
                std_val = final_results[f"std_{metric}"]
                ci_val = final_results[f"ci_{metric}"]
                tqdm.write(f"  {metric}: {mean_val:.4f} ± {std_val:.4f} (95% CI: ±{ci_val:.4f})")

        # 任务扩充统计
        if self.task_sampler and "mean_augmentation_ratio" in final_results:
            aug_ratio = final_results["mean_augmentation_ratio"]
            tqdm.write(f"  augmentation_ratio: {aug_ratio:.2f}x")

    def _save_results(self, final_results, all_results):
        """保存结果到文件"""
        import json

        # 保存最终统计结果
        final_results_path = os.path.join(self.save_dir, "final_results.json")
        with open(final_results_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            json_compatible_results = {}
            for k, v in final_results.items():
                if isinstance(v, np.ndarray):
                    json_compatible_results[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_compatible_results[k] = v.item()
                else:
                    json_compatible_results[k] = v
            json.dump(json_compatible_results, f, indent=2)

        # 保存每个任务的详细结果
        if self.config.meta_test.evaluation.save_predictions:
            detailed_results_path = os.path.join(self.save_dir, "detailed_results.json")
            with open(detailed_results_path, "w") as f:
                json_compatible_all_results = []
                for result in all_results:
                    json_compatible_result = {}
                    for k, v in result.items():
                        if isinstance(v, (np.integer, np.floating)):
                            json_compatible_result[k] = v.item()
                        else:
                            json_compatible_result[k] = v
                    json_compatible_all_results.append(json_compatible_result)
                json.dump(json_compatible_all_results, f, indent=2)

        tqdm.write(f"Results saved to: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Meta-test Trainer with Optional Task Augmentation"
    )
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VAE检查点路径")
    parser.add_argument("--score_checkpoint", type=str, help="Score检查点路径（任务扩充可选）")
    args = parser.parse_args()

    # 创建训练器并开始测试
    trainer = MetaTestTrainer(args.config, args.vae_checkpoint, args.score_checkpoint)
    results = trainer.run_meta_test()

    print(f"Meta-test completed. Results saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
