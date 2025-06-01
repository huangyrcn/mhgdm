import os
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import ml_collections
import wandb
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch
import models.Encoders as Encoders
import models.Decoders as Decoders
from models.Decoders import FermiDiracDecoder, Classifier
from utils.graph_utils import node_flags


class OneHot_CrossEntropy(torch.nn.Module):
    """One-hot cross entropy loss for graph reconstruction"""
    def __init__(self):
        super(OneHot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=2)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.sum(torch.sum(loss, dim=2))
        return loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self._setup_wandb()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_checkpoints()

    def _setup_wandb(self):
        # 检查是否禁用wandb
        no_wandb = getattr(self.config.wandb, 'no_wandb', False)
        debug_mode = getattr(self.config, 'debug', False)
        
        # 如果no_wandb为True或者debug模式，则禁用wandb
        if no_wandb or debug_mode:
            mode = "disabled"
        else:
            mode = "online" if self.config.wandb.online else "offline"
            
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.run_name,
            config=self.config.to_dict(),
            settings=wandb.Settings(_disable_stats=True),
            mode=mode,
            dir=os.path.join("logs", "wandb"),
        )

    def _setup_device(self):
        self.seed = load_seed(self.config.seed)
        device_obj, device_str = load_device(
            device_setting=getattr(self.config, "device", "auto"),
            device_count=getattr(self.config, "device_count", 1)
        )
        self.device = device_obj
        self.config.device = device_str

    def _setup_data(self):
        self.dataset = MyDataset(self.config.data, self.config.fsl_task)
        self.train_loader, self.test_loader = self.dataset.get_loaders()

    def _setup_model(self):
        from utils.loader import load_model
        self.encoder = load_model(self.config.encoder).to(self.device)
        self.manifold = getattr(self.encoder, "manifold", None)
        
        # Setup decoder for graph reconstruction
        self.use_decoder = getattr(self.config, 'use_decoder', False)
        if self.use_decoder and hasattr(self.config, 'decoder'):
            # Load decoder configuration and model
            decoder_config = self.config.decoder
            # Update decoder config to match encoder output
            if hasattr(self.encoder, 'dim'):
                decoder_config.max_feat_num = self.encoder.dim
            elif hasattr(self.config.encoder, 'dim'):
                decoder_config.max_feat_num = self.config.encoder.dim
            
            self.decoder = load_model(decoder_config).to(self.device)
            self.reconstruction_loss_fn = OneHot_CrossEntropy()
            print(f"Decoder initialized: {decoder_config.model_class}")
        else:
            self.decoder = None
            self.reconstruction_loss_fn = None
        
        # Edge prediction setup
        self.do_edge_pred = self.config.encoder.pred_edge
        if self.do_edge_pred:
            self.edge_predictor = FermiDiracDecoder(manifold=self.manifold).to(self.device)
            self.edge_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        
        # Collect all parameters for optimizer
        params = list(self.encoder.parameters())
        if self.decoder is not None:
            params.extend(list(self.decoder.parameters()))
        if self.do_edge_pred:
            params.extend(list(self.edge_predictor.parameters()))
            
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.train.lr, weight_decay=self.config.train.get("weight_decay", 0)
        )

    def _setup_checkpoints(self):
        self.save_dir = f"./checkpoints/{self.config.run_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "best.pth")
        
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, 'w') as f:
            # Convert ml_collections.ConfigDict to dict for OmegaConf
            config_dict = self.config.to_dict()
            OmegaConf.save(config=config_dict, f=f)

    def _save_checkpoint(self, test_acc, epoch):
        checkpoint_data = {
            "encoder_state_dict": self.encoder.state_dict(),
            "encoder_config": self.config.encoder.to_dict(),
            "test_accuracy": test_acc,
            "epoch": epoch,
        }
        if self.decoder is not None:
            checkpoint_data["decoder_state_dict"] = self.decoder.state_dict()
            checkpoint_data["decoder_config"] = self.config.decoder.to_dict()
        if self.do_edge_pred:
            checkpoint_data["edge_predictor_state_dict"] = self.edge_predictor.state_dict()
        torch.save(checkpoint_data, self.save_path)
        # 使用 tqdm.write 确保输出不与进度条冲突
        tqdm.write(f"Saved best model: {test_acc:.3f} -> {self.save_path}")

    def _get_embeddings(self, x, adj):
        mask = node_flags(adj).unsqueeze(-1)
        if "node_mask" in self.encoder.forward.__code__.co_varnames:
            posterior = self.encoder(x, adj, mask)
        else:
            posterior = self.encoder(x, adj)
        return posterior

    def _compute_reconstruction_loss(self, posterior, x, adj):
        """Compute graph reconstruction loss - following HVAE implementation"""
        if self.decoder is None or self.reconstruction_loss_fn is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Following HVAE: use sample() for reconstruction, not mode()
        h = posterior.sample() if hasattr(posterior, 'sample') else posterior
        
        # Get node mask - following HVAE format
        node_mask = node_flags(adj)
        node_mask = node_mask.unsqueeze(-1)
        
        # Decode to reconstruct node features - following HVAE decoder call
        if "node_mask" in self.decoder.forward.__code__.co_varnames:
            type_pred = self.decoder(h, adj, node_mask)
        else:
            type_pred = self.decoder(h, adj)
        
        # Apply mask and compute reconstruction loss - following HVAE exactly
        # HVAE: node_classification_loss = self.loss_fn(type_pred * node_mask, x)
        type_pred_masked = type_pred * node_mask
        recon_loss = self.reconstruction_loss_fn(type_pred_masked, x)
        
        # Following HVAE: normalize by node_mask.sum() at the end
        return recon_loss / node_mask.sum()

    def _compute_kl_loss(self, posterior):
        """Compute KL divergence loss - following HVAE implementation"""
        if not hasattr(posterior, 'kl'):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Following HVAE: kl = posterior.kl(), then kl.sum() / node_mask.sum()
        kl = posterior.kl()
        return kl

    def _edge_prediction_loss(self, posterior, adj):
        if not self.do_edge_pred:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get point estimate for edge prediction
        z = posterior.mode() if hasattr(posterior, 'mode') else posterior
            
        edge_pred = self.edge_predictor(z)  # (B, N, N, 4)
        adj_flat = torch.triu(adj, 1).long().view(-1)
        pred_flat = edge_pred.view(-1, 4)
        
        # Simple random sampling of edges
        pos_idx = torch.where(adj_flat == 1)[0]
        neg_idx = torch.where(adj_flat == 0)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        n_samples = min(len(pos_idx), len(neg_idx), 1000)
        pos_sample = pos_idx[torch.randperm(len(pos_idx))[:n_samples]]
        neg_sample = neg_idx[torch.randperm(len(neg_idx))[:n_samples]]
        sample_idx = torch.cat([pos_sample, neg_sample])
        
        return self.edge_loss_fn(pred_flat[sample_idx], adj_flat[sample_idx])

    def train_one_step(self, batch):
        self.optimizer.zero_grad()
        x, adj, _ = load_batch(batch, self.device)
        
        # Get encoder posterior
        posterior = self._get_embeddings(x, adj)
        
        # Get node mask for proper normalization (following HVAE)
        node_mask = node_flags(adj).unsqueeze(-1)
        
        # Compute all losses
        recon_loss = self._compute_reconstruction_loss(posterior, x, adj)
        
        # Following HVAE: kl.sum() / node_mask.sum()
        kl_raw = self._compute_kl_loss(posterior)
        if hasattr(kl_raw, 'sum'):
            kl_loss = kl_raw.sum() / node_mask.sum()
        else:
            kl_loss = kl_raw / node_mask.sum()
            
        edge_loss = self._edge_prediction_loss(posterior, adj)
        
        # Combine losses with weights from config
        loss_weights = getattr(self.config.train, 'loss_weights', {})
        recon_weight = loss_weights.get('reconstruction', 1.0)
        kl_weight = loss_weights.get('kl', 0.1)
        edge_weight = loss_weights.get('edge', 1.0)
        
        total_loss = (recon_weight * recon_loss + 
                     kl_weight * kl_loss + 
                     edge_weight * edge_loss)
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'edge_loss': edge_loss.item()
        }

    def train(self):
        print(f"Starting training: {self.config.run_name}")
        best_acc = 0.0
        eval_interval = getattr(self.config.train, "eval_interval", 10)
        
        # 创建外层训练进度条
        training_pbar = tqdm(range(self.config.train.num_epochs), desc="Training")
        
        for epoch in training_pbar:
            self.encoder.train()
            if self.decoder is not None:
                self.decoder.train()
            if self.do_edge_pred:
                self.edge_predictor.train()
                
            total_losses = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'edge_loss': 0}
            batch_count = 0
            
            # 添加实时损失显示的进度条
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in pbar:
                losses = self.train_one_step(batch)
                for key in total_losses:
                    total_losses[key] += losses[key]
                batch_count += 1
                
                # 更新进度条显示当前损失
                pbar.set_postfix({
                    'Total': f'{losses["total_loss"]:.4f}',
                    'Recon': f'{losses["recon_loss"]:.4f}',
                    'KL': f'{losses["kl_loss"]:.4f}',
                    'Edge': f'{losses["edge_loss"]:.4f}'
                })
            
            # Calculate average losses
            avg_losses = {key: total_losses[key] / len(self.train_loader) for key in total_losses}
            avg_loss = avg_losses['total_loss']  # For backward compatibility
            
            metrics = {"epoch": epoch}
            metrics.update({f"train_{key}": value for key, value in avg_losses.items()})
            
            # 更新外层进度条显示当前epoch的损失
            training_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{self.config.train.num_epochs}',
                'Loss': f'{avg_loss:.4f}'
            })
            
            # Evaluation logic
            should_eval = (eval_interval == 0 and epoch == self.config.train.num_epochs - 1) or \
                         (eval_interval > 0 and ((epoch + 1) % eval_interval == 0 or epoch == self.config.train.num_epochs - 1))
            
            if should_eval:
                train_acc, train_std, train_loss = self.eval("train")
                test_acc, test_std, test_loss = self.eval("test")
                
                metrics.update({
                    "train_accuracy": train_acc, "train_std": train_std, "train_eval_loss": train_loss,
                    "test_accuracy": test_acc, "test_std": test_std, "test_loss": test_loss,
                })
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    self._save_checkpoint(test_acc, epoch + 1)
                    
                # 更新外层进度条显示评估结果
                training_pbar.set_postfix({
                    'Epoch': f'{epoch+1}/{self.config.train.num_epochs}',
                    'Loss': f'{avg_loss:.4f}',
                    'TestAcc': f'{test_acc:.3f}'
                })
                    
                # 确保在新行输出评估结果，避免与进度条混淆
                tqdm.write(f"Epoch {epoch+1}: Train {train_acc:.3f}±{train_std:.3f}, Test {test_acc:.3f}±{test_std:.3f}")
            
            wandb.log(metrics)
            
        print("\nTraining completed.")
        return self.save_path

    def _get_embeddings_for_task(self, x, adj):
        with torch.no_grad():
            posterior = self._get_embeddings(x, adj)
            
            # Following HVAE mean_graph computation exactly
            emb_from_posterior = posterior.mode() if hasattr(posterior, "mode") else posterior
            
            # Handle dimension consistency
            if emb_from_posterior.dim() == 2:
                emb_for_pooling = emb_from_posterior.unsqueeze(1)
            else:
                emb_for_pooling = emb_from_posterior
            
            # Transform to tangent space if using manifold (following HVAE)
            if self.manifold is not None:
                emb_in_tangent_space = self.manifold.logmap0(emb_for_pooling)
            else:
                emb_in_tangent_space = emb_for_pooling
            
            # Pool features (following HVAE)
            mean_pooled_features = emb_in_tangent_space.mean(dim=1)
            max_pooled_features = emb_in_tangent_space.max(dim=1).values
            mean_graph = torch.cat([mean_pooled_features, max_pooled_features], dim=-1)
            
            return mean_graph

    def eval_one_step(self, task):
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_label = task["support_set"]["label"].to(self.device)
        query_x = task["query_set"]["x"].to(self.device)
        query_adj = task["query_set"]["adj"].to(self.device)
        query_label = task["query_set"]["label"].to(self.device)

        support_emb = self._get_embeddings_for_task(support_x, support_adj)
        query_emb = self._get_embeddings_for_task(query_x, query_adj)

        n_way = len(torch.unique(support_label))
        classifier = Classifier(
            model_dim=support_emb.shape[-1] // 2,  # Original dim before concat
            classifier_dropout=0.0,
            classifier_bias=True,
            manifold=None,
            n_classes=n_way,
        ).to(self.device)

        # Train classifier
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        classifier.train()
        for _ in range(50):  # Fixed epochs
            optimizer.zero_grad()
            logits = classifier.decode(support_emb, adj=None)
            loss = loss_fn(logits, support_label)
            loss.backward()
            optimizer.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            query_logits = classifier.decode(query_emb, adj=None)
            query_loss = loss_fn(query_logits, query_label)
            preds = torch.argmax(query_logits, dim=1)
            accuracy = (preds == query_label).float().mean().item()

        return accuracy, query_loss.item()

    def eval(self, mode="test", ckpt_path=None):
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])

        self.encoder.eval()
        is_train = (mode == "train")
        config = self.config.fsl_task
        
        accuracies = []
        losses = []

        if is_train:
            # Fixed number of tasks for train evaluation
            n_tasks = self.config.train.get("num_eval_tasks", 100)
            for _ in tqdm(range(n_tasks), desc=f"Eval {mode}", leave=False):
                task = self.dataset.sample_one_task(
                    is_train=is_train, N_way=config.N_way, K_shot=config.K_shot, R_query=config.R_query
                )
                if task:
                    acc, loss = self.eval_one_step(task)
                    accuracies.append(acc)
                    losses.append(loss)
        else:
            # Test mode - iterate through all test data
            start_idx = 0
            max_idx = len(self.dataset.test_graphs) - config.K_shot * self.dataset.num_test_classes_remapped
            
            with tqdm(desc=f"Eval {mode}", leave=False) as pbar:
                while start_idx < max_idx:
                    task = self.dataset.sample_one_task(
                        is_train=is_train, N_way=config.N_way, K_shot=config.K_shot, 
                        R_query=config.R_query, query_pool_start_index=start_idx
                    )
                    if not task:
                        break
                        
                    acc, loss = self.eval_one_step(task)
                    accuracies.append(acc)
                    losses.append(loss)
                    start_idx += config.N_way * config.R_query
                    pbar.update(1)

        if not accuracies:
            return 0.0, 0.0, 0.0
            
        return np.mean(accuracies), np.std(accuracies), np.mean(losses)


@hydra.main(config_path="configs", config_name="train_encoder_hvae", version_base="1.3")
def main(cfg: DictConfig):
    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    print(f"Training: {config.run_name}")
    
    trainer = Trainer(config)
    best_ckpt_path = trainer.train()
    
    # 确保完整路径显示
    print(f"\nTraining completed successfully!")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Config saved to: {os.path.dirname(best_ckpt_path)}/config.yaml")


if __name__ == "__main__":
    main()
