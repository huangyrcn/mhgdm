"""
Meta-Learning Test Script
基于预训练的encoder和score model训练分类头进行元学习测试
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import trange, tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加numpy到torch.load的安全全局变量中
torch.serialization.add_safe_globals([
    np.generic, np.ndarray,
    np.bool_, np.int_,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float_, 
    np.float16, np.float32, np.float64,
    np.complex_,
    np.complex64, np.complex128,
    np.object_, np.str_, np.bytes_
])

# 导入必要的模块
import models.Encoders as Encoders
from models.Decoders import Classifier
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch
from utils.graph_utils import node_flags
from utils.manifolds_utils import get_manifold
from utils.protos_utils import compute_protos_from
import ml_collections
from sampler import Sampler

class MetaTestTrainer:
    """
    元学习测试训练器，基于预训练的encoder和可选的score model进行分类头训练
    """
    
    def __init__(self, config):
        self.config = config
        self.run_name = config.run_name
        
        # 初始化wandb
        mode = (
            "disabled"
            if self.config.debug
            else ("online" if self.config.wandb.online else "offline")
        )
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.run_name,
            config=OmegaConf.to_container(self.config, resolve=True),
            settings=wandb.Settings(_disable_stats=True),
            mode=mode,
            dir=os.path.join("logs", "wandb"),
        )
        
        # 设置随机种子和设备
        self.seed = load_seed(self.config.seed)
        device_obj, device_str = load_device(
            device_setting=getattr(self.config, "device", "auto"),
            device_count=getattr(self.config, "device_count", 1)
        )
        self.device = device_obj
        self.config.device = device_str
        
        # 加载数据集
        self.dataset = MyDataset(self.config.data, self.config.fsl_task)
        self.train_loader, self.test_loader = self.dataset.get_loaders()
        
        # 加载预训练的编码器
        self._load_encoder()
        
        # 可选：加载预训练的分数模型用于数据增强
        self.sampler = None
        if self.config.data_augmentation.use_score_model:
            self._load_score_model()
        
        # 设置元测试参数
        self.num_test_tasks = self.config.meta_test.num_test_tasks
        self.classifier_epochs = self.config.meta_test.classifier.epochs
        self.classifier_lr = self.config.meta_test.classifier.lr
        self.classifier_patience = self.config.meta_test.classifier.patience
        
        # 创建保存目录
        self.save_dir = f"./checkpoints/{self.config.data.name}/meta_test/{self.run_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"MetaTestTrainer initialized:")
        print(f"  - Encoder: {self.encoder.__class__.__name__}")
        print(f"  - Device: {self.device}")
        print(f"  - Manifold: {self.manifold}")
        print(f"  - Num test tasks: {self.num_test_tasks}")
        if self.sampler:
            print(f"  - Data augmentation: enabled (k={self.config.data_augmentation.k_augment})")
        else:
            print(f"  - Data augmentation: disabled")
    
    def _load_encoder(self):
        """加载预训练的编码器"""
        encoder_path = self.config.model_paths.encoder_ckpt_path
        if not encoder_path or not os.path.exists(encoder_path):
            raise ValueError(f"Encoder checkpoint path not found: {encoder_path}")
        
        print(f"Loading encoder from: {encoder_path}")
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
        
        # 解析checkpoint格式
        if "encoder_config" in checkpoint:
            # 新格式
            encoder_config_dict = checkpoint["encoder_config"]
            encoder_state_dict = checkpoint["encoder_state_dict"]
            encoder_name = encoder_config_dict["model_class"]
        elif "model_config" in checkpoint:
            # 旧格式
            encoder_checkpoint_config_dict = checkpoint["model_config"]
            encoder_state_dict = checkpoint["encoder_state_dict"]
            encoder_name = encoder_checkpoint_config_dict.model.encoder
            encoder_config_dict = encoder_checkpoint_config_dict.encoder.to_dict()
        else:
            raise ValueError(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}")
        
        # 创建编码器
        import ml_collections
        from utils.loader import load_model
        encoder_config_for_instantiation = ml_collections.ConfigDict(encoder_config_dict)
        self.encoder = load_model(encoder_config_for_instantiation).to(self.device)
        self.encoder.load_state_dict(encoder_state_dict)
        
        # 设置为评估模式
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        
        # 获取流形
        self.manifold = getattr(self.encoder, "manifold", None)
        self.encoder_output_dim = encoder_config_dict.get('dim', 64)
        
        print(f"Encoder loaded: {encoder_name}, output_dim: {self.encoder_output_dim}")
        if self.manifold:
            print(f"Encoder manifold: {self.manifold}")
    
    def _load_score_model(self):
        """加载预训练的分数模型用于数据增强"""
        try:
            # 动态检查采样器是否可用
            from sampler import Sampler
            sampler_available = True
        except ImportError:
            print("Sampler module not available. Cannot load score model for data augmentation.")
            self.config.data_augmentation.use_score_model = False
            return
            
        score_path = self.config.model_paths.score_ckpt_path
        if not score_path or not os.path.exists(score_path):
            print(f"Warning: Score model path not found: {score_path}. Disabling data augmentation.")
            self.config.data_augmentation.use_score_model = False
            return
        
        print(f"Loading score model for data augmentation from: {score_path}")
        
        # 创建采样器配置
        from ml_collections import ConfigDict
        sampler_config = ConfigDict({
            'score_ckpt_path': score_path,
            'encoder_ckpt_path': self.config.model_paths.encoder_ckpt_path,
            'data': self.config.data,
            'fsl_task': self.config.fsl_task,
            'exp_name': 'meta_test_augment',
            'sampler': self.config.sampler,
            'sample': self.config.sample,
            'device': self.config.device,
            'dataloader': None  # 将在sampler中创建
        })
        sampler_config.sample.k_augment = self.config.data_augmentation.k_augment
        
        try:
            self.sampler = Sampler(sampler_config)
            print("Score model loaded for data augmentation")
        except Exception as e:
            print(f"Failed to load score model: {e}")
            print("Disabling data augmentation")
            self.config.data_augmentation.use_score_model = False
            self.sampler = None
    
    def get_embeddings(self, x, adj):
        """获取图的嵌入向量"""
        mask = node_flags(adj)
        node_mask_for_encoder = mask.unsqueeze(-1)
        
        with torch.no_grad():
            # 检查编码器forward函数是否需要node_mask参数
            if "node_mask" in self.encoder.forward.__code__.co_varnames:
                z = self.encoder(x, adj, node_mask_for_encoder)
            else:
                z = self.encoder(x, adj)
            
            # 提取嵌入向量
            if hasattr(z, "mode"):
                z = z.mode()
            
            # 转换双曲嵌入到欧几里得空间（如果需要）
            if self.manifold and hasattr(self.manifold, 'logmap0'):
                z = self.manifold.logmap0(z)
            
            # 应用池化策略
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
                raise ValueError(f"Unsupported pooling method: {pooling_method}")
            
            # 可选的标准化
            if self.config.meta_test.embedding.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def train_classifier_on_task(self, task):
        """在单个任务上训练分类头"""
        # 获取支持集和查询集数据
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_label = task["support_set"]["label"].to(self.device)
        
        query_x = task["query_set"]["x"].to(self.device)
        query_adj = task["query_set"]["adj"].to(self.device)
        query_label = task["query_set"]["label"].to(self.device)
        
        # 获取嵌入向量
        support_emb = self.get_embeddings(support_x, support_adj)
        query_emb = self.get_embeddings(query_x, query_adj)
        
        # 计算类别数和嵌入维度
        n_way = len(torch.unique(support_label))
        emb_dim = support_emb.shape[-1]
        
        # Classifier期望的输入维度是 model_dim，而实际输入是 model_dim * pool_factor
        # 根据池化方法确定输入维度因子
        pooling_method = self.config.meta_test.embedding.pooling_method
        if pooling_method == "mean_max":
            model_dim_for_classifier = self.encoder_output_dim  # 因为mean_max会产生2倍维度
        else:
            model_dim_for_classifier = emb_dim  # 使用实际嵌入维度
        
        # 创建分类器
        classifier = Classifier(
            model_dim=model_dim_for_classifier,
            classifier_dropout=self.config.meta_test.classifier.dropout,
            classifier_bias=self.config.meta_test.classifier.bias,
            manifold=None,  # 分类器在欧几里得空间工作
            n_classes=n_way,
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(
            classifier.parameters(), 
            lr=self.classifier_lr,
            weight_decay=self.config.meta_test.classifier.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        
        # 训练分类器
        best_loss = float("inf")
        patience_counter = 0
        
        classifier.train()
        for epoch in range(self.classifier_epochs):
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
                if patience_counter >= self.classifier_patience:
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
            
            f1 = f1_score(query_label_np, predicted_np, average='macro', zero_division=0)
            precision = precision_score(query_label_np, predicted_np, average='macro', zero_division=0)
            recall = recall_score(query_label_np, predicted_np, average='macro', zero_division=0)
        
        # 可选：支持集评估
        support_metrics = {}
        if self.config.meta_test.eval_support:
            with torch.no_grad():
                support_logits = classifier(support_emb)
                support_loss = loss_fn(support_logits, support_label)
                _, support_predicted = torch.max(support_logits, 1)
                support_accuracy = (support_predicted == support_label).float().mean().item()
                
                support_metrics = {
                    'support_loss': support_loss.item(),
                    'support_accuracy': support_accuracy
                }
        
        return {
            'query_loss': query_loss.item(),
            'query_accuracy': accuracy,
            'query_f1': f1,
            'query_precision': precision,
            'query_recall': recall,
            **support_metrics
        }
    
    def run_meta_test(self):
        """运行元学习测试"""
        print(f"\n{'='*50}")
        print(f"Starting meta-learning test: {self.run_name}")
        print(f"{'='*50}\n")
        
        # 数据增强（如果启用）
        test_loader = self.test_loader
        if self.config.data_augmentation.use_score_model and self.sampler:
            print("Performing data augmentation...")
            # 设置sampler的dataloader
            self.sampler.dataloader = self.test_loader
            augmented_result = self.sampler.sample(need_eval=False)
            if hasattr(augmented_result, '__iter__') and hasattr(augmented_result, 'dataset'):
                test_loader = augmented_result
                print(f"Data augmentation completed. New dataset size: {len(test_loader.dataset)}")
        
        # 生成测试任务
        print(f"Generating {self.num_test_tasks} test tasks...")
        n_way = self.config.fsl_task.N_way
        k_shot = self.config.fsl_task.K_shot
        r_query = self.config.fsl_task.R_query
        
        # 存储结果
        all_results = []
        
        # 进度条
        pbar = tqdm(range(self.num_test_tasks), desc="Meta-testing")
        
        for task_idx in pbar:
            # 生成单个任务
            try:
                # 计算查询池起始索引，确保不会超出范围
                query_pool_size = len(self.dataset.deterministic_test_query_pool_indices)
                max_start_index = max(0, query_pool_size - r_query)
                
                if task_idx * r_query > max_start_index:
                    # 如果超出范围，随机选择起始位置
                    query_start_idx = np.random.randint(0, max_start_index + 1) if max_start_index >= 0 else 0
                else:
                    query_start_idx = min(task_idx * r_query, max_start_index)
                
                task = self.dataset.sample_one_task(
                    is_train=False,
                    N_way=n_way,
                    K_shot=k_shot,
                    R_query=r_query,
                    query_pool_start_index=query_start_idx
                )
            except Exception as e:
                print(f"Warning: Failed to generate task {task_idx}: {e}")
                # 尝试随机生成
                try:
                    query_start_idx = np.random.randint(0, max_start_index + 1) if max_start_index >= 0 else 0
                    task = self.dataset.sample_one_task(
                        is_train=False,
                        N_way=n_way,
                        K_shot=k_shot,
                        R_query=r_query,
                        query_pool_start_index=query_start_idx
                    )
                except Exception as e2:
                    print(f"Error: Cannot generate task {task_idx}: {e2}")
                    continue
            
            # 训练分类头并评估
            try:
                task_result = self.train_classifier_on_task(task)
                all_results.append(task_result)
                
                # 更新进度条
                if len(all_results) > 0:
                    mean_acc = np.mean([r['query_accuracy'] for r in all_results])
                    pbar.set_postfix({'mean_acc': f'{mean_acc:.4f}'})
                
                # 定期记录到wandb
                if (task_idx + 1) % self.config.logging.log_interval == 0:
                    self._log_intermediate_results(all_results, task_idx + 1)
                    
            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue
        
        # 计算最终统计结果
        final_results = self._compute_final_statistics(all_results)
        
        # 记录最终结果
        self._log_final_results(final_results)
        
        # 保存结果
        self._save_results(final_results, all_results)
        
        print(f"\nMeta-learning test completed!")
        print(f"Final Results:")
        for metric, value in final_results.items():
            if 'mean' in metric or 'std' in metric:
                print(f"  {metric}: {value:.4f}")
        
        wandb.finish()
        return final_results
    
    def _log_intermediate_results(self, results, task_count):
        """记录中间结果到wandb"""
        if len(results) == 0:
            return
        
        # 计算当前的平均指标
        current_stats = {}
        for metric in ['query_accuracy', 'query_f1', 'query_precision', 'query_recall']:
            values = [r[metric] for r in results if metric in r]
            if values:
                current_stats[f'running_mean_{metric}'] = np.mean(values)
                current_stats[f'running_std_{metric}'] = np.std(values)
        
        current_stats['completed_tasks'] = task_count
        wandb.log(current_stats)
    
    def _compute_final_statistics(self, results):
        """计算最终统计结果"""
        if len(results) == 0:
            return {}
        
        final_stats = {}
        confidence_level = self.config.evaluation.confidence_interval
        
        # 计算每个指标的统计值
        for metric in ['query_accuracy', 'query_f1', 'query_precision', 'query_recall', 
                      'query_loss', 'support_accuracy', 'support_loss']:
            values = [r[metric] for r in results if metric in r]
            if values:
                values = np.array(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # 计算置信区间
                confidence_interval = std_val * 1.96 / np.sqrt(len(values))  # 95% CI
                
                final_stats[f'mean_{metric}'] = mean_val
                final_stats[f'std_{metric}'] = std_val
                final_stats[f'ci_{metric}'] = confidence_interval
                final_stats[f'num_samples_{metric}'] = len(values)
        
        final_stats['total_tasks'] = len(results)
        return final_stats
    
    def _log_final_results(self, final_results):
        """记录最终结果到wandb"""
        wandb.log(final_results, commit=True)
        
        # 创建结果总结表格
        summary_data = []
        for metric in ['query_accuracy', 'query_f1', 'query_precision', 'query_recall']:
            if f'mean_{metric}' in final_results:
                summary_data.append([
                    metric,
                    final_results[f'mean_{metric}'],
                    final_results[f'std_{metric}'],
                    final_results[f'ci_{metric}']
                ])
        
        if summary_data:
            table = wandb.Table(
                columns=['Metric', 'Mean', 'Std', '95% CI'],
                data=summary_data
            )
            wandb.log({'final_results_table': table})
    
    def _save_results(self, final_results, all_results):
        """保存结果到文件"""
        import json
        
        # 保存最终统计结果
        final_results_path = os.path.join(self.save_dir, 'final_results.json')
        with open(final_results_path, 'w') as f:
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
        if self.config.evaluation.save_predictions:
            detailed_results_path = os.path.join(self.save_dir, 'detailed_results.json')
            with open(detailed_results_path, 'w') as f:
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
        
        print(f"Results saved to: {self.save_dir}")


@hydra.main(config_path="configs", config_name="meta_test_poincare_proto", version_base="1.3")
def main(cfg: DictConfig):
    """主函数：启动元学习测试"""
    
    # 检查必要的路径配置
    if not cfg.model_paths.encoder_ckpt_path:
        raise ValueError("encoder_ckpt_path must be provided in config.model_paths")
    
    if cfg.data_augmentation.use_score_model and not cfg.model_paths.score_ckpt_path:
        raise ValueError("score_ckpt_path must be provided when data_augmentation.use_score_model=True")
    
    # 创建训练器并运行测试
    trainer = MetaTestTrainer(cfg)
    results = trainer.run_meta_test()
    
    return results


if __name__ == "__main__":
    main()
