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
from utils.loader import load_seed, load_device, load_batch, load_model
from models.HVAE import HVAE
from models.Decoders import Classifier
from utils.graph_utils import node_flags


class HVAETrainer:
    def __init__(self, config):
        self.config = config
        self._setup_wandb()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_checkpoints()

    def _setup_wandb(self):
        mode = "disabled" if self.config.debug else ("online" if self.config.wandb.online else "offline")
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
        # Load encoder and decoder models to get their classes
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder
        
        # Update input_feat_dim based on actual dataset feature dimension
        actual_feat_dim = self.dataset.max_feat_num
        if hasattr(encoder_config, 'input_feat_dim'):
            encoder_config.input_feat_dim = actual_feat_dim
        if hasattr(decoder_config, 'input_feat_dim'):
            decoder_config.input_feat_dim = actual_feat_dim
        
        # Load models to get classes
        encoder_model = load_model(encoder_config)
        decoder_model = load_model(decoder_config)
        
        # Extract HVAE-specific parameters from config
        hvae_config = getattr(self.config, 'hvae', {})
        
        # Prepare encoder parameters
        encoder_params = encoder_config.to_dict()
        # Remove parameters that are not constructor arguments for the encoder
        params_to_remove = ['model_class', 'max_feat_num', 'pred_edge', 'use_centroid', 'input_manifold']
        for param in params_to_remove:
            if param in encoder_params:
                del encoder_params[param]
            
        # Prepare decoder parameters
        decoder_params = decoder_config.to_dict()
        # Remove parameters that are not constructor arguments for the decoder
        for param in params_to_remove:
            if param in decoder_params:
                del decoder_params[param]
        
        # Initialize HVAE model
        self.hvae = HVAE(
            device=self.device,
            encoder_class=encoder_model.__class__,
            encoder_params=encoder_params,
            decoder_class=decoder_model.__class__,
            decoder_params=decoder_params,
            manifold_type=encoder_config.get('manifold', 'Euclidean'),
            train_class_num=self.dataset.num_train_classes_remapped,
            dim=encoder_config.dim,
            pred_node_class=hvae_config.get('pred_node_class', True),
            use_kl_loss=hvae_config.get('use_kl_loss', True),
            use_base_proto_loss=hvae_config.get('use_base_proto_loss', True),
            use_sep_proto_loss=hvae_config.get('use_sep_proto_loss', True),
            pred_edge=hvae_config.get('pred_edge', False),
            pred_graph_class=hvae_config.get('pred_graph_class', False),
            classifier_dropout=hvae_config.get('classifier_dropout', 0.0),
            classifier_bias=hvae_config.get('classifier_bias', True),
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.hvae.parameters(), 
            lr=self.config.train.lr, 
            weight_decay=self.config.train.get("weight_decay", 0)
        )

    def _setup_checkpoints(self):
        self.save_dir = f"./checkpoints/{self.config.run_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "best.pth")
        
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, 'w') as f:
            config_dict = self.config.to_dict()
            OmegaConf.save(config=config_dict, f=f)

    def _save_checkpoint(self, test_acc, epoch):
        checkpoint_data = {
            "hvae_state_dict": self.hvae.state_dict(),
            "hvae_config": {
                "encoder_config": self.config.encoder.to_dict(),
                "decoder_config": self.config.decoder.to_dict(),
                "hvae_config": getattr(self.config, 'hvae', {})
            },
            "test_accuracy": test_acc,
            "epoch": epoch,
        }
        torch.save(checkpoint_data, self.save_path)
        tqdm.write(f"Saved best HVAE model: {test_acc:.3f} -> {self.save_path}")

    def _get_graph_embeddings(self, x, adj):
        """Get graph-level embeddings for few-shot learning evaluation"""
        with torch.no_grad():
            node_mask = node_flags(adj).unsqueeze(-1)
            
            # Get posterior from encoder
            posterior = self.hvae.encoder(x, adj, node_mask)
            
            # Get mode (point estimate) from posterior
            emb_from_posterior = posterior.mode() if hasattr(posterior, "mode") else posterior
            
            # Handle dimension consistency
            if emb_from_posterior.dim() == 2:
                emb_for_pooling = emb_from_posterior.unsqueeze(1)
            else:
                emb_for_pooling = emb_from_posterior
            
            # Transform to tangent space if using manifold
            if self.hvae.manifold is not None:
                emb_in_tangent_space = self.hvae.manifold.logmap0(emb_for_pooling)
            else:
                emb_in_tangent_space = emb_for_pooling
            
            # Pool features (mean and max pooling)
            mean_pooled_features = emb_in_tangent_space.mean(dim=1)
            max_pooled_features = emb_in_tangent_space.max(dim=1).values
            mean_graph = torch.cat([mean_pooled_features, max_pooled_features], dim=-1)
            
            return mean_graph

    def train_one_step(self, batch):
        self.optimizer.zero_grad()
        x, adj, graph_labels = load_batch(batch, self.device)
        
        # Get all losses from HVAE forward pass
        (node_classification_loss, kl_loss, edge_loss, 
         base_proto_loss, sep_proto_loss, graph_classification_loss, 
         graph_classification_acc) = self.hvae(x, adj, graph_labels)
        
        # Get loss weights from config
        loss_weights = getattr(self.config.train, 'loss_weights', {})
        node_weight = loss_weights.get('node_classification', 1.0)
        kl_weight = loss_weights.get('kl', 0.1)
        edge_weight = loss_weights.get('edge', 1.0)
        base_proto_weight = loss_weights.get('base_proto', 1.0)
        sep_proto_weight = loss_weights.get('sep_proto', 0.1)
        graph_class_weight = loss_weights.get('graph_classification', 1.0)
        
        # Combine all losses
        total_loss = (node_weight * node_classification_loss + 
                     kl_weight * kl_loss + 
                     edge_weight * edge_loss +
                     base_proto_weight * base_proto_loss +
                     sep_proto_weight * sep_proto_loss +
                     graph_class_weight * graph_classification_loss)
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'node_classification_loss': node_classification_loss.item(),
            'kl_loss': kl_loss.item(),
            'edge_loss': edge_loss.item(),
            'base_proto_loss': base_proto_loss.item(),
            'sep_proto_loss': sep_proto_loss.item(),
            'graph_classification_loss': graph_classification_loss.item(),
            'graph_classification_acc': graph_classification_acc
        }

    def train(self):
        print(f"Starting HVAE training: {self.config.run_name}")
        best_acc = 0.0
        eval_interval = getattr(self.config.train, "eval_interval", 10)
        
        # Create outer training progress bar
        training_pbar = tqdm(range(self.config.train.num_epochs), desc="HVAE Training")
        
        for epoch in training_pbar:
            self.hvae.train()
            self.hvae.current_epoch = epoch  # Update epoch for potential scheduling
                
            total_losses = {
                'total_loss': 0, 'node_classification_loss': 0, 'kl_loss': 0, 
                'edge_loss': 0, 'base_proto_loss': 0, 'sep_proto_loss': 0,
                'graph_classification_loss': 0, 'graph_classification_acc': 0
            }
            batch_count = 0
            
            # Add real-time loss display progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in pbar:
                losses = self.train_one_step(batch)
                for key in total_losses:
                    total_losses[key] += losses[key]
                batch_count += 1
                
                # Update progress bar display with current losses
                pbar.set_postfix({
                    'Total': f'{losses["total_loss"]:.4f}',
                    'Node': f'{losses["node_classification_loss"]:.4f}',
                    'KL': f'{losses["kl_loss"]:.4f}',
                    'Proto': f'{losses["base_proto_loss"]:.4f}',
                    'GraphAcc': f'{losses["graph_classification_acc"]:.3f}'
                })
            
            # Calculate average losses
            avg_losses = {key: total_losses[key] / len(self.train_loader) for key in total_losses}
            avg_loss = avg_losses['total_loss']
            
            metrics = {"epoch": epoch}
            metrics.update({f"train_{key}": value for key, value in avg_losses.items()})
            
            # Update outer progress bar display
            training_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{self.config.train.num_epochs}',
                'Loss': f'{avg_loss:.4f}',
                'GraphAcc': f'{avg_losses["graph_classification_acc"]:.3f}'
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
                    
                # Update outer progress bar display with evaluation results
                training_pbar.set_postfix({
                    'Epoch': f'{epoch+1}/{self.config.train.num_epochs}',
                    'Loss': f'{avg_loss:.4f}',
                    'TestAcc': f'{test_acc:.3f}',
                    'GraphAcc': f'{avg_losses["graph_classification_acc"]:.3f}'
                })
                    
                # Output evaluation results on new line to avoid confusion with progress bar
                tqdm.write(f"Epoch {epoch+1}: Train {train_acc:.3f}±{train_std:.3f}, Test {test_acc:.3f}±{test_std:.3f}")
            
            wandb.log(metrics)
            
        print("\nHVAE training completed.")
        return self.save_path

    def eval_one_step(self, task):
        """Evaluate one few-shot learning task"""
        support_x = task["support_set"]["x"].to(self.device)
        support_adj = task["support_set"]["adj"].to(self.device)
        support_label = task["support_set"]["label"].to(self.device)
        query_x = task["query_set"]["x"].to(self.device)
        query_adj = task["query_set"]["adj"].to(self.device)
        query_label = task["query_set"]["label"].to(self.device)

        # Get graph embeddings using HVAE encoder
        support_emb = self._get_graph_embeddings(support_x, support_adj)
        query_emb = self._get_graph_embeddings(query_x, query_adj)

        n_way = len(torch.unique(support_label))
        classifier = Classifier(
            model_dim=support_emb.shape[-1] // 2,  # Original dim before concat
            classifier_dropout=0.0,
            classifier_bias=True,
            manifold=None,
            n_classes=n_way,
        ).to(self.device)

        # Train classifier on support set
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        classifier.train()
        for _ in range(50):  # Fixed epochs for classifier training
            optimizer.zero_grad()
            logits = classifier.decode(support_emb, adj=None)
            loss = loss_fn(logits, support_label)
            loss.backward()
            optimizer.step()

        # Evaluate on query set
        classifier.eval()
        with torch.no_grad():
            query_logits = classifier.decode(query_emb, adj=None)
            query_loss = loss_fn(query_logits, query_label)
            preds = torch.argmax(query_logits, dim=1)
            accuracy = (preds == query_label).float().mean().item()

        return accuracy, query_loss.item()

    def eval(self, mode="test", ckpt_path=None):
        """Evaluate HVAE on few-shot learning tasks"""
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.hvae.load_state_dict(checkpoint["hvae_state_dict"])

        self.hvae.eval()
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


@hydra.main(config_path="configs", config_name="train_hvae", version_base="1.3")
def main(cfg: DictConfig):
    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    print(f"HVAE Training: {config.run_name}")
    
    trainer = HVAETrainer(config)
    best_ckpt_path = trainer.train()
    
    # Ensure complete path display
    print(f"\nHVAE training completed successfully!")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Config saved to: {os.path.dirname(best_ckpt_path)}/config.yaml")


if __name__ == "__main__":
    main()
