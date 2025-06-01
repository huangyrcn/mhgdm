import os
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import ml_collections
import wandb
import numpy as np
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy

# Add numpy to safe globals for torch.load with weights_only=True
torch.serialization.add_safe_globals([
    numpy.generic, numpy.ndarray, numpy.bool_, numpy.int_,
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
    numpy.float_, numpy.float16, numpy.float32, numpy.float64,
    numpy.complex_, numpy.complex64, numpy.complex128,
    numpy.object_, numpy.str_, numpy.bytes_
])

from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch, load_model_optimizer, load_ema, load_loss_fn
from utils.graph_utils import node_flags
import models.Encoders as Encoders
from utils.manifolds_utils import get_manifold
from utils.protos_utils import compute_protos_from


class ScoreTrainer:
    def __init__(self, config):
        self.config = config
        self._setup_wandb()
        self._setup_device()
        self._setup_data()
        self._load_encoder()
        self._setup_models()
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
            dir=os.path.join("logs", "wandb_score"),
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

    def _load_encoder(self):
        encoder_path = self.config.encoder_path
        print(f"Loading encoder from: {encoder_path}")
        
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
        
        if "encoder_config" in checkpoint:
            encoder_config_dict = checkpoint["encoder_config"]
            encoder_state_dict = checkpoint["encoder_state_dict"]
            encoder_name = encoder_config_dict["model_class"]
        elif "model_config" in checkpoint:
            encoder_checkpoint_config_dict = checkpoint["model_config"]
            encoder_state_dict = checkpoint["encoder_state_dict"]
            encoder_name = encoder_checkpoint_config_dict.model.encoder
            encoder_config_dict = encoder_checkpoint_config_dict.encoder.to_dict()
        else:
            raise ValueError(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}")

        print(f"Loading encoder: {encoder_name}")
        
        from ml_collections import ConfigDict
        from utils.loader import load_model
        encoder_config = ConfigDict(encoder_config_dict)
        
        self.encoder = load_model(encoder_config).to(self.device)
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        
        # Setup manifold
        self.manifold = getattr(self.encoder, "manifold", None)
        print(f"Using manifold from encoder: {self.manifold}")
        
        # Store encoder output dimension and manifold parameters
        self.encoder_output_dim = encoder_config_dict.get('dim', 64)
        encoder_manifold = encoder_config_dict.get('manifold', 'PoincareBall')
        encoder_c = encoder_config_dict.get('c', 1.0)
        
        # Update score model configs based on encoder configuration
        print(f"Updating score model configs with encoder parameters:")
        print(f"  - dim: {self.encoder_output_dim}")
        print(f"  - manifold: {encoder_manifold}")
        print(f"  - c: {encoder_c}")
        
        # Update score model parameters based on encoder and data configuration
        # Update score_model_x
        self.config.score_model_x.max_feat_num = self.encoder_output_dim
        self.config.score_model_x.max_node_num = self.config.data.max_node_num
        self.config.score_model_x.manifold = encoder_manifold
        self.config.score_model_x.c = encoder_c
        
        # Update score_model_a
        self.config.score_model_a.max_feat_num = self.encoder_output_dim
        self.config.score_model_a.max_node_num = self.config.data.max_node_num
        self.config.score_model_a.manifold = encoder_manifold
        self.config.score_model_a.c = encoder_c

        self.params_x = self.config.score_model_x.to_dict()



        self.params_adj = self.config.score_model_a.to_dict()

    def _setup_models(self):
        # It is assumed that self.config.model.x and self.config.model.adj are always present
        # and correctly populated from the main configuration files.
        # Fallback ConfigDict creation and parameter defaulting have been removed as per simplification.

        # Load models and optimizers using config objects
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(
            device=self.device, manifold=self.manifold,
            lr=self.config.train.lr, weight_decay=self.config.train.weight_decay,
            lr_schedule=self.config.train.lr_schedule, lr_decay=self.config.train.lr_decay,
            config=self.config.score_model_x,  # Access using the name from defaults list
        )
        
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(
            device=self.device, manifold=self.manifold,
            lr=self.config.train.lr, weight_decay=self.config.train.weight_decay,
            lr_schedule=self.config.train.lr_schedule, lr_decay=self.config.train.lr_decay,
            config=self.config.score_model_a, # Access using the name from defaults list
        )

        # Setup EMA
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)
        
        # Compute prototypes
        self.protos_train = compute_protos_from(self.encoder, self.train_loader, self.device)
        self.protos_test = compute_protos_from(self.encoder, self.test_loader, self.device)

        # Setup loss function
        self.loss_fn = load_loss_fn(
            sde_x_config=self.config.sde.x, sde_adj_config=self.config.sde.adj,
            reduce_mean=self.config.train.reduce_mean, eps=self.config.train.eps,
            manifold=self.manifold, encoder=self.encoder,
        )

        total_params = sum(p.numel() for p in self.model_x.parameters()) + sum(p.numel() for p in self.model_adj.parameters())
        print(f"Score network parameters: {total_params / 1e6:.2f}M")

    def _setup_checkpoints(self):
        self.save_dir = f"./checkpoints/{self.config.run_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "best.pth")
        
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, 'w') as f:
            config_dict = self.config.to_dict()
            OmegaConf.save(config=config_dict, f=f)

    def _save_checkpoint(self, test_loss, epoch):
        save_dict = {
            "epoch": epoch,
            "model_config": self.config.to_dict(),
            "params_x": self.params_x,
            "params_adj": self.params_adj,
            "x_state_dict": self.model_x.state_dict(),
            "adj_state_dict": self.model_adj.state_dict(),
            "ema_x_state_dict": self.ema_x.state_dict(),
            "ema_adj_state_dict": self.ema_adj.state_dict(),
            "optimizer_x_state_dict": self.optimizer_x.state_dict(),
            "optimizer_adj_state_dict": self.optimizer_adj.state_dict(),
            "scheduler_x_state_dict": self.scheduler_x.state_dict() if self.scheduler_x else None,
            "scheduler_adj_state_dict": self.scheduler_adj.state_dict() if self.scheduler_adj else None,
            "test_loss": test_loss,
        }
        torch.save(save_dict, self.save_path)
        tqdm.write(f"Saved best model: {test_loss:.4f} -> {self.save_path}")

    def train_one_step(self, batch, debug_mode=False):
        x, adj, labels = load_batch(batch, self.device)
        
       
        loss_x, loss_adj = self.loss_fn(
            self.model_x, self.model_adj, x, adj, labels, self.protos_train
        )
        
        self.optimizer_x.zero_grad()
        self.optimizer_adj.zero_grad()
        loss_x.backward()
        loss_adj.backward()

        self.optimizer_x.step()
        self.optimizer_adj.step()

        self.ema_x.update(self.model_x.parameters())
        self.ema_adj.update(self.model_adj.parameters())

        return loss_x.item(), loss_adj.item()

    def eval_one_epoch(self):
        self.model_x.eval()
        self.model_adj.eval()
        
        test_x_losses = []
        test_adj_losses = []

        with torch.no_grad():
            self.ema_x.store(self.model_x.parameters())
            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.store(self.model_adj.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                x, adj, labels = load_batch(batch, self.device)
                
                loss_x, loss_adj = self.loss_fn(
                    self.model_x, self.model_adj, x, adj, labels, self.protos_test
                )
                test_x_losses.append(loss_x.item())
                test_adj_losses.append(loss_adj.item())

            self.ema_x.restore(self.model_x.parameters())
            self.ema_adj.restore(self.model_adj.parameters())
            
        return np.mean(test_x_losses), np.mean(test_adj_losses)

    def train(self):
        print(f"Starting score training: {self.config.run_name}")
        
       
        self.model_x.train()
        self.model_adj.train()
        

        
        best_test_loss = float('inf')
        eval_interval = getattr(self.config.train, "eval_interval", 50)
        
        # 创建外层训练进度条
        training_pbar = tqdm(range(self.config.train.num_epochs), desc="Training")
        
        for epoch in training_pbar:
            self.model_x.train()
            self.model_adj.train()
            
            train_x_losses = []
            train_adj_losses = []
            
            # Create progress bar with postfix for real-time loss display
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
            
            for batch in pbar:
                loss_x, loss_adj = self.train_one_step(batch)
                train_x_losses.append(loss_x)
                train_adj_losses.append(loss_adj)
                
                # Update progress bar with current loss values
                current_avg_x = np.mean(train_x_losses)
                current_avg_adj = np.mean(train_adj_losses)
                total_loss = current_avg_x + current_avg_adj
                
                pbar.set_postfix({
                    'X': f'{loss_x:.4f}',
                    'Adj': f'{loss_adj:.4f}',
                    'Total': f'{total_loss:.4f}'
                })
            
            avg_train_x = np.mean(train_x_losses)
            avg_train_adj = np.mean(train_adj_losses)
            train_total_loss = avg_train_x + avg_train_adj
            
            # 更新外层进度条显示当前epoch的损失
            training_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{self.config.train.num_epochs}',
                'Loss': f'{train_total_loss:.4f}'
            })
            
            metrics = {
                "epoch": epoch,
                "train_loss_x": avg_train_x,
                "train_loss_adj": avg_train_adj,
                "train_total_loss": train_total_loss,
            }
            
            # Evaluation logic
            should_eval = (eval_interval == 0 and epoch == self.config.train.num_epochs - 1) or \
                         (eval_interval > 0 and ((epoch + 1) % eval_interval == 0 or epoch == self.config.train.num_epochs - 1))
            
            if should_eval:
                test_x, test_adj = self.eval_one_epoch()
                total_test_loss = test_x + test_adj
                
                metrics.update({
                    "test_loss_x": test_x,
                    "test_loss_adj": test_adj,
                    "test_total_loss": total_test_loss,
                })
                
                if total_test_loss < best_test_loss:
                    best_test_loss = total_test_loss
                    self._save_checkpoint(total_test_loss, epoch + 1)
                
                # 更新外层进度条显示评估结果
                training_pbar.set_postfix({
                    'Epoch': f'{epoch+1}/{self.config.train.num_epochs}',
                    'Loss': f'{train_total_loss:.4f}',
                    'TestLoss': f'{total_test_loss:.4f}'
                })
                    
                tqdm.write(f"Epoch {epoch+1}: Train {train_total_loss:.4f}, Test {total_test_loss:.4f}")
            
            # Update learning rate
            if self.config.train.lr_schedule:
                if self.scheduler_x:
                    self.scheduler_x.step()
                if self.scheduler_adj:
                    self.scheduler_adj.step()
            
            wandb.log(metrics)
            
        print("\nTraining completed.")
        return self.save_path


@hydra.main(config_path="configs", config_name="train_score_poincare", version_base="1.3")
def main(cfg: DictConfig):
    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    print(f"Training: {config.run_name}")
    
    trainer = ScoreTrainer(config)
    best_ckpt_path = trainer.train()
    
    print(f"\nTraining completed successfully!")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Config saved to: {os.path.dirname(best_ckpt_path)}/config.yaml")


if __name__ == "__main__":
    main()
