import os
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import ml_collections
import wandb
import numpy as np
import math
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch
import models.Encoders as Encoders
# Use the FermiDiracDecoder from models.Decoders as it's defined in HVAE.py
from models.HVAE import FermiDiracDecoder
from models.Decoders import Classifier
from utils.graph_utils import node_flags # Ensure this utility is available and correct


class Trainer:
    def __init__(self, config):
        self.config = config
        self.run_name = config.run_name
        mode = (
            "disabled"
            if self.config.debug
            else ("online" if self.config.wandb.online else "offline")
        )
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.run_name,
            config=self.config.to_dict(),
            settings=wandb.Settings(_disable_stats=True),
            mode=mode,
            dir=os.path.join("logs", "wandb"),
        )
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.dataset = MyDataset(self.config)
        self.train_loader, self.test_loader = self.dataset.get_loaders()

        # For evaluation
        self.num_eval_tasks = self.config.train.get(
            "num_eval_tasks", 100
        )  # Number of tasks for evaluation
        # Use names consistent with configs/fsl_task/10shot.yaml
        self.classifier_epochs = self.config.fsl_task.get("epochs_head", 50)
        self.classifier_lr = self.config.fsl_task.get("lr_head", 0.01)
        self.classifier_patience = self.config.fsl_task.get("head_train_patience", 10)

    
        # Checkpoint saving logic
        self.save_dir = f"./checkpoints/{self.config.data.name}/{self.run_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file_path = os.path.join(self.save_dir, "best.pth")

        # 初始化编码器

        encoder_class_to_instantiate = getattr(Encoders, config.model.encoder)

        # Pass the full config object to the encoder, not just config.model
        self.encoder = encoder_class_to_instantiate(config).to(self.device)


        # 获取编码器的流形（如果有）
        self.manifold = getattr(self.encoder, "manifold", None)

        # 检查配置中是否启用了边预测
        self.do_edge_prediction = self.config.model.pred_edge

        # 条件初始化边预测器
        if self.do_edge_prediction:
            # 初始化边预测器 - 使用 HVAE.py 中的 FermiDiracDecoder 定义
            # 它只需要 manifold 作为参数
            if self.manifold is None:
                # Fallback or error if manifold is required but not available
                # For now, let's assume Euclidean if no specific manifold from encoder
                # This part might need adjustment based on how FermiDiracDecoder handles None manifold
                print("WARNING: Manifold is None, edge_predictor might not work as expected if it relies on a specific manifold.")
            self.edge_predictor = FermiDiracDecoder(manifold=self.manifold).to(self.device)

            # 损失函数 - FermiDiracDecoder 输出多类别概率，使用交叉熵损失
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean") # 与 HVAE.py 保持一致

        # 优化器
        if self.do_edge_prediction:
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.edge_predictor.parameters()),
                lr=self.config.train.lr,
                weight_decay=self.config.train.get("weight_decay", 0),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.get("weight_decay", 0),
            )

    def train_one_step(self, batch):
        # torch.autograd.set_detect_anomaly(True) # Moved to train() method for broader scope

        self.optimizer.zero_grad()
        x, adj, labels = load_batch(batch, self.device)
        
        mask = node_flags(adj) # (B,N)
        edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1) # (B,N,N)
        node_mask_for_encoder = mask.unsqueeze(-1) # (B,N,1)


        if "node_mask" in self.encoder.forward.__code__.co_varnames:
            z_posterior = self.encoder(x, adj, node_mask_for_encoder)
        else:
            z_posterior = self.encoder(x, adj)
        
        if hasattr(z_posterior, "mode"):
            z = z_posterior.mode()   
        else:
            z = z_posterior


        current_loss_value = -1 # Placeholder for actual loss

        if self.do_edge_prediction:
            # Edge prediction logic aligned with HVAE.py
            edge_pred_output = self.edge_predictor(z) # (B, N, N, 4)
            
            triu_mask_for_pred = torch.triu(edge_mask, 1).unsqueeze(-1) # (B, N, N, 1) to broadcast with (B,N,N,4)
            
            edge_pred = edge_pred_output * triu_mask_for_pred # Apply mask
            edge_pred = edge_pred.view(-1, 4)    # Reshape for loss, 4 for edge types

          
            adj_for_loss = torch.triu(adj, 1).long().view(-1) # Flattened upper triangle of adj

          
            triu_mask_flat_numpy = triu_mask_for_pred.squeeze(-1).view(-1).cpu().numpy().astype(bool)
            
            adj_numpy = adj_for_loss.cpu().numpy() # Already (potentially) boolean due to .long() then comparison, but ensure type for bitwise ops
            adj_numpy_bool = adj_numpy.astype(bool)
            adj_numpy_invert = ~adj_numpy_bool & triu_mask_flat_numpy
            
            pos_edges = np.where(adj_numpy_bool & triu_mask_flat_numpy)[0] # Positive edges within the valid mask
            neg_edges = np.where(adj_numpy_invert)[0] # Negative edges (non-edges) within the valid mask

            if len(pos_edges) == 0 or len(neg_edges) == 0:

                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                choise_num = np.min([len(pos_edges), len(neg_edges)])

                pos_id_indices = np.random.choice(len(pos_edges), choise_num)
                neg_id_indices = np.random.choice(len(neg_edges), choise_num)

                pos_id = pos_edges[pos_id_indices]
                neg_id = neg_edges[neg_id_indices]
                
                choose_id = torch.tensor(np.append(pos_id, neg_id), device=edge_pred.device, dtype=torch.long)
                
                selected_logits = edge_pred[choose_id]
                selected_targets = adj_for_loss[choose_id]

                if selected_logits.nelement() > 0:
                    # Removed DEBUG print for selected_logits
                    pass
                
                if selected_targets.nelement() > 0:
                    if not (selected_targets.min() >= 0 and selected_targets.max() < 4): # 检查目标是否在 [0, C-1] 范围内
                         print(f"DEBUG: selected_targets out of range [0, 3] for CrossEntropyLoss with 4 classes. Min: {selected_targets.min()}, Max: {selected_targets.max()}. Clamping or error handling might be needed if adj can have values other than 0 or 1 for edge types.")
                         # This indicates adj might not be just 0/1 for no_edge/edge but actual types.
                         # If adj contains edge types (0 for no edge, 1, 2, 3 for bond types), this is fine.
                         # The HVAE snippet implies adj is long and used directly.
                    loss = self.loss_fn(selected_logits, selected_targets)
            current_loss_value = loss.item()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            current_loss_value = loss.item()

   
        loss.backward()

      

        self.optimizer.step()

        return loss.item()

    def train(self):
        torch.autograd.set_detect_anomaly(True) # Enable anomaly detection
        print(f"\n{'-'*50}")
        print(f"Starting training: {self.run_name}")
        print(f"{'-'*50}\n")

        best_test_accuracy = 0.0  # Initialize best test accuracy

        t = trange(self.config.train.num_epochs, desc="[Epoch]", leave=True)
        for epoch in t:
            self.encoder.train()
            if self.do_edge_prediction:
                self.edge_predictor.train()
            total_loss = 0
            num_batches = 0
            pbar = tqdm(self.train_loader, desc="Training", leave=False)
            for batch in pbar:
                loss = self.train_one_step(batch)
                total_loss += loss
                num_batches += 1
                pbar.set_description(f"Loss: {loss:.4f}")
            avg_loss = total_loss / num_batches
            t.set_description(
                f"[Epoch {epoch+1}/{self.config.train.num_epochs}] Loss: {avg_loss:.4f}"
            )

            # 确保每个epoch都向wandb提交数据
            metrics_to_log = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "current_lr": self.optimizer.param_groups[0]["lr"],
            }

            # 按eval_interval评估，使用配置中的值，默认为10
            eval_interval = getattr(self.config.train, "eval_interval", 10)
            # 如果eval_interval为0，则只在最后一次评估
            if (eval_interval == 0 and (epoch == self.config.train.num_epochs - 1)) or \
               (eval_interval != 0 and (epoch + 1) % eval_interval == 0) or \
               (epoch == self.config.train.num_epochs - 1):

                print(f"\n[Eval] Epoch {epoch+1}")

                # 计算训练准确率
                train_acc, train_std, train_loss = self.eval(mode="train")
                print(f"Train Accuracy: {train_acc:.4f} ± {train_std:.4f}, Loss: {train_loss:.4f}")

                # 计算测试准确率
                test_acc, test_std, test_loss = self.eval(mode="test")
                print(f"Test Accuracy: {test_acc:.4f} ± {test_std:.4f}, Loss: {test_loss:.4f}")

                # 添加评估指标到metrics_to_log字典中
                metrics_to_log.update(
                    {
                        "train_accuracy": train_acc,
                        "train_std": train_std,
                        "train_eval_loss": train_loss,
                        "test_accuracy": test_acc,
                        "test_std": test_std,
                        "test_loss": test_loss,
                    }
                )

                if test_acc > best_test_accuracy:
                    best_test_accuracy = test_acc       
                    checkpoint_data = {
                        "epoch": epoch + 1,
                        "model_config": self.config.to_dict(),
                        "encoder_state_dict": self.encoder.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_test_accuracy": best_test_accuracy,
                    }
                    if self.do_edge_prediction:
                        checkpoint_data["edge_predictor_state_dict"] = self.edge_predictor.state_dict()
                    
                    torch.save(checkpoint_data, self.save_file_path)
                    print(f"Epoch {epoch+1}: Saved new best model with test accuracy: {best_test_accuracy:.4f} to {self.save_file_path}")

            # 每个epoch结束时提交一次数据
            wandb.log(metrics_to_log, commit=True)
        print("Training completed.")


        return self.save_file_path

    def eval_one_step(self, task):
        """
        Evaluate the encoder on a single task
        Args:
            task: A dictionary containing support_set and query_set
        Returns:
            accuracy: Accuracy on the query set
            loss: Loss on the query set
        """
        # Get support and query data
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_label = task["support_set"]["label"].to(self.device)

        query_x = task["query_set"]["x"].to(self.device)
        query_adj = task["query_set"]["adj"].to(self.device)
        query_label = task["query_set"]["label"].to(self.device)

        # Create masks
        support_mask = node_flags(support_adj)
        query_mask = node_flags(query_adj)

        # Get embeddings from encoder
        with torch.no_grad():
            # Support set embedding
            if "node_mask" in self.encoder.forward.__code__.co_varnames:
                support_z = self.encoder(support_x, support_adj, support_mask.unsqueeze(-1))
            else:
                support_z = self.encoder(support_x, support_adj)

            # Query set embedding
            if "node_mask" in self.encoder.forward.__code__.co_varnames:
                query_z = self.encoder(query_x, query_adj, query_mask.unsqueeze(-1))
            else:
                query_z = self.encoder(query_x, query_adj)

        # Extract embedding vectors
        if hasattr(support_z, "mode"):
            support_z = support_z.mode()
        if hasattr(query_z, "mode"):
            query_z = query_z.mode()

        # Transform hyperbolic embeddings to Euclidean if needed
        if self.manifold:
            support_z = self.manifold.logmap0(support_z)
            query_z = self.manifold.logmap0(query_z)

        # Prepare embeddings for classifier input
        # (mean pooling + max pooling concatenation, similar to trainer.py)
        support_mean = support_z.mean(dim=1)
        support_max = support_z.max(dim=1).values
        support_emb = torch.cat([support_mean, support_max], dim=-1)

        query_mean = query_z.mean(dim=1)
        query_max = query_z.max(dim=1).values
        query_emb = torch.cat([query_mean, query_max], dim=-1)

        # Initialize and train a classifier
        n_way = len(torch.unique(support_label))
        original_dim = support_z.shape[-1]  # 原始嵌入维度
        classifier = Classifier(
            model_dim=original_dim,
            classifier_dropout=0.0,
            classifier_bias=True,
            manifold=None,  # Euclidean
            n_classes=n_way,
        ).to(self.device)

        # Train classifier on support set
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.classifier_lr)
        loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        patience_counter = 0

        classifier.train()
        for _ in range(self.classifier_epochs):
            optimizer.zero_grad()
            logits = classifier.decode(support_emb, adj=None)
            loss = loss_fn(logits, support_label)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.classifier_patience:
                break

        # Evaluate on query set
        classifier.eval()
        with torch.no_grad():
            query_logits = classifier.decode(query_emb, adj=None)
            query_loss = loss_fn(query_logits, query_label)
            preds = torch.argmax(query_logits, dim=1)
            accuracy = (preds == query_label).float().mean().item()

        return accuracy, query_loss.item()

    def eval(self, mode="test", ckpt_path=None):
        """
        Evaluate the encoder on multiple tasks
        Args:
            mode: 'train' or 'test'
            ckpt_path: Optional path to a checkpoint file to load before evaluation.
        Returns:
            mean_acc: Mean accuracy across tasks
            std_acc: Standard deviation of accuracy
            mean_loss: Mean loss across tasks
        """
        if ckpt_path:
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.encoder.load_state_dict(checkpoint["encoder_state_dict"])

        self.encoder.eval()  # Set encoder to evaluation mode

        is_train =( mode == "train")
        n_way = self.config.fsl_task.N_way
        k_shot = self.config.fsl_task.K_shot
        r_query = self.config.fsl_task.R_query

        accuracies = []
        losses = []

        # 对于train模式，使用固定数量的随机任务
        if is_train:
            n_tasks = self.num_eval_tasks
            for _ in tqdm(range(n_tasks), desc=f"Evaluating on {mode} set", leave=False):
                task = self.dataset.sample_one_task(
                    is_train=is_train, N_way=n_way, K_shot=k_shot, R_query=r_query
                )

                if task is None:
                    continue

                accuracy, loss = self.eval_one_step(task)
                accuracies.append(accuracy)
                losses.append(loss)

        # 对于test模式，使用start_test_idx控制循环
        else:
            # 初始化起始索引
            start_test_idx = 0

            # 估计总任务数用于进度条
            max_idx = len(self.dataset.test_graphs) - k_shot * self.dataset.num_test_classes_remapped
            estimated_tasks = (
                max(1, math.ceil(max_idx / (n_way * r_query))) if (n_way * r_query) > 0 else 10
            )

            # 按照参考代码的循环条件迭代
            with tqdm(total=estimated_tasks, desc=f"Evaluating on {mode} set", leave=False) as pbar:
                while (
                    start_test_idx
                    < len(self.dataset.test_graphs) - k_shot * self.dataset.num_test_classes_remapped
                ):
                    # 采样任务 - 传递 start_test_idx
                    task = self.dataset.sample_one_task(
                        is_train=is_train, N_way=n_way, K_shot=k_shot, R_query=r_query, query_pool_start_index=start_test_idx
                    )

                    if task is None:
                        break

                    accuracy, loss = self.eval_one_step(task)
                    accuracies.append(accuracy)
                    losses.append(loss)

                    # 按照参考代码递增索引
                    start_test_idx += n_way * r_query
                    pbar.update(1)

        # 计算统计结果
        if not accuracies:
            print(f"WARNING: No tasks were evaluated for {mode} mode")
            return 0.0, 0.0, 0.0

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_loss = np.mean(losses)

        print(f"Evaluated {len(accuracies)} tasks, Mean acc: {mean_acc:.4f}, Std: {std_acc:.4f}")

        return mean_acc, std_acc, mean_loss


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="configs", config_name="train_encoder", version_base="1.3")
    def main(cfg: DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        config = ml_collections.ConfigDict(cfg_dict)
        print("\n===== 当前训练配置(config) =====")
        print(config)
        print("===== 配置输出结束 =====\n")
        trainer = Trainer(config)
        best_ckpt_path = trainer.train() # train() now returns the best checkpoint path or None

        print(f"\nEvaluating on train set using checkpoint: {best_ckpt_path}")
        train_acc, train_std, train_loss = trainer.eval(mode="train", ckpt_path=best_ckpt_path)
        print(f"Train Accuracy (from ckpt): {train_acc:.4f} ± {train_std:.4f}, Loss: {train_loss:.4f}")

        # 评估测试集性能
        print(f"\nEvaluating on test set using checkpoint: {best_ckpt_path}")
        test_acc, test_std, test_loss = trainer.eval(mode="test", ckpt_path=best_ckpt_path)
        print(f"Test Accuracy (from ckpt): {test_acc:.4f} ± {test_std:.4f}, Loss: {test_loss:.4f}")

    main()
