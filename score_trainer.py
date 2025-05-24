import os
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import ml_collections
import wandb
import numpy as np
import time
# Add numpy to safe globals for torch.load with weights_only=True
import numpy
torch.serialization.add_safe_globals([
    numpy.generic, numpy.ndarray,
    numpy.bool_, numpy.int_,
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
    numpy.float_, 
    numpy.float16, numpy.float32, numpy.float64,
    numpy.complex_,
    numpy.complex64, numpy.complex128,
    numpy.object_, numpy.str_, numpy.bytes_ # Adding object, str, bytes as well
])

from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch, load_model_optimizer, load_ema, load_loss_fn
# from models.HVAE import HVAE #确保移除，除非HVAE在score_trainer其他地方用到
import models.Encoders as Encoders # 用于直接加载编码器
from utils.manifolds_utils import get_manifold
from utils.protos_utils import compute_protos_from


class ScoreTrainer:
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
            dir=os.path.join("logs", "wandb_score"), 
        )
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.dataset = MyDataset(self.config)
        self.train_loader, self.test_loader = self.dataset.get_loaders()

        # 修改：强制要求必须提供并加载编码器
        encoder_path = getattr(self.config.model, 'encoder_path', None)

        print(f"Loading encoder from checkpoint: {encoder_path}")
        checkpoint = torch.load(encoder_path, map_location=self.config.device, weights_only=False)

        encoder_checkpoint_config_dict = checkpoint["model_config"]
        encoder_state_dict = checkpoint["encoder_state_dict"]

        encoder_config_for_instantiation = encoder_checkpoint_config_dict

        encoder_name = encoder_config_for_instantiation.model.encoder 
        EncoderClass = getattr(Encoders, encoder_name)
        
        self.encoder = EncoderClass(encoder_config_for_instantiation).to(self.device)
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.requires_grad_(False)
        self.encoder = self.encoder.eval()
        
        # 检查流形一致性
        encoder_manifold_obj = getattr(self.encoder, "manifold", None)
        # 从当前score_trainer的配置中获取预期的流形设置
        current_config_manifold_name = self.config.model.manifold
        current_config_manifold_c = self.config.model.c
        expected_config_manifold = get_manifold(current_config_manifold_name, current_config_manifold_c)
        
        # 如果编码器有流形，则检查与当前score_trainer配置的流形是否一致
        if type(encoder_manifold_obj) != type(expected_config_manifold):
            raise ValueError(
                f"Manifold type mismatch! Encoder '{encoder_name}' was trained with manifold type: {type(encoder_manifold_obj)}, "
                f"but current score_trainer config specifies: {type(expected_config_manifold)}. "
                f"Please ensure the manifold configuration in score_trainer matches the encoder's training configuration."
            )
        
        # 检查曲率是否匹配 (如果流形有曲率属性 'c')
        if hasattr(encoder_manifold_obj, 'c') and hasattr(expected_config_manifold, 'c'):
            # 注意：直接比较浮点数可能不精确，但对于配置参数通常是固定的
            if abs(encoder_manifold_obj.c.item() - expected_config_manifold.c.item()) > 1e-6 : # encoder_manifold_obj.c 是 tensor
                raise ValueError(
                    f"Manifold curvature mismatch! Encoder '{encoder_name}' was trained with curvature: {encoder_manifold_obj.c.item()}, "
                    f"but current score_trainer config specifies: {expected_config_manifold.c.item()}. "
                    f"Please ensure the curvature matches the encoder's training configuration."
                )
        
        # 如果检查通过，使用编码器的流形
        self.manifold = encoder_manifold_obj
        print(f"Encoder '{encoder_name}' loaded. Using encoder's manifold: {self.manifold}")

        print(f"Final manifold for score training: {self.manifold}")
        
        self.params_x = self.config.model.x.to_dict()
        self.params_x["manifold"] = self.manifold
        self.params_adj = self.config.model.adj.to_dict()
        self.params_adj["manifold"] = self.manifold

        (
            self.model_x,
            self.optimizer_x,
            self.scheduler_x,
        ) = load_model_optimizer(
            self.config,
            self.params_x,
        )
        (
            self.model_adj,
            self.optimizer_adj,
            self.scheduler_adj,
        ) = load_model_optimizer(self.config, self.params_adj)

        total_params = sum(
            [param.nelement() for param in self.model_x.parameters()]
            + [param.nelement() for param in self.model_adj.parameters()]
        )
        print(f"Number of parameters for score networks: {total_params / 1e6:.2f}M")

        self.ema_x = load_ema(
            self.model_x,
            decay=self.config.train.ema,
        )
        self.ema_adj = load_ema(
            self.model_adj,
            decay=self.config.train.ema,
        )
        
        # Compute protos
        self.protos_train = compute_protos_from(
            self.encoder,
            self.train_loader,
            self.device,
        )
        self.protos_test = compute_protos_from(
            self.encoder,
            self.test_loader,
            self.device,
        )

        self.loss_fn = load_loss_fn(
            self.config,
            self.manifold,
            encoder=self.encoder,
        )
        
        self.save_dir = f"./checkpoints/{self.config.data.name}/{self.config.exp_name}/{self.config.timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)


    def train_one_epoch(self, epoch):
        self.model_x.train()
        self.model_adj.train()

        train_x_losses = []
        train_adj_losses = []

        for _, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)):
            x, adj, labels = load_batch(batch, self.device)

            loss_x, loss_adj = self.loss_fn(
                self.model_x,
                self.model_adj,
                x,
                adj,
                labels,
                self.protos_train,
            )

            if torch.isnan(loss_x) or torch.isnan(loss_adj):
                raise ValueError("NaN loss encountered during training")

            self.optimizer_x.zero_grad()
            self.optimizer_adj.zero_grad()
            loss_x.backward()
            loss_adj.backward()

            if self.config.train.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_x.parameters(),
                    self.config.train.grad_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model_adj.parameters(),
                    self.config.train.grad_norm,
                )

            self.optimizer_x.step()
            self.optimizer_adj.step()

            self.ema_x.update(self.model_x.parameters())
            self.ema_adj.update(self.model_adj.parameters())

            train_x_losses.append(loss_x.item())
            train_adj_losses.append(loss_adj.item())

        if self.config.train.lr_schedule:
            if self.scheduler_x:
                self.scheduler_x.step()
            if self.scheduler_adj:
                self.scheduler_adj.step()
        
        return np.mean(train_x_losses), np.mean(train_adj_losses)

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

            for _, batch in enumerate(tqdm(self.test_loader, desc="Evaluating", leave=False)):
                x, adj, labels = load_batch(batch, self.device)
                
                loss_x, loss_adj = self.loss_fn(
                    self.model_x,
                    self.model_adj,
                    x,
                    adj,
                    labels,
                    self.protos_test,
                )
                test_x_losses.append(loss_x.item())
                test_adj_losses.append(loss_adj.item())

            self.ema_x.restore(self.model_x.parameters())
            self.ema_adj.restore(self.model_adj.parameters())
            
        return np.mean(test_x_losses), np.mean(test_adj_losses)

    def train(self):
        print(f"\n{'='*50}")
        print(f"Starting score network training: {self.run_name}")
        print(f"{'='*50}\n")

        best_mean_test_loss = float('inf')

        for epoch in trange(self.config.train.num_epochs, desc="[Overall Progress]", leave=True):
            t_start = time.time()

            mean_train_x_loss, mean_train_adj_loss = self.train_one_epoch(epoch)
            mean_test_x_loss, mean_test_adj_loss = self.eval_one_epoch()
            
            total_test_loss = mean_test_x_loss + mean_test_adj_loss
            epoch_time = time.time() - t_start

            log_metrics = {
                "epoch": epoch + 1,
                "train_loss_x": mean_train_x_loss,
                "train_loss_adj": mean_train_adj_loss,
                "test_loss_x": mean_test_x_loss,
                "test_loss_adj": mean_test_adj_loss,
                "total_test_loss": total_test_loss,
                "epoch_time_seconds": epoch_time,
                "lr_x": self.optimizer_x.param_groups[0]['lr'] if self.optimizer_x else 0,
                "lr_adj": self.optimizer_adj.param_groups[0]['lr'] if self.optimizer_adj else 0,
            }
            wandb.log(log_metrics, commit=True)

            if (epoch + 1) % self.config.train.print_interval == 0 or epoch == self.config.train.num_epochs - 1:
                tqdm.write(
                    f"[Epoch {epoch+1}/{self.config.train.num_epochs}] "
                    f"Train X: {mean_train_x_loss:.4f}, Train Adj: {mean_train_adj_loss:.4f} | "
                    f"Test X: {mean_test_x_loss:.4f}, Test Adj: {mean_test_adj_loss:.4f} | "
                    f"Total Test Loss: {total_test_loss:.4f} | Time: {epoch_time:.2f}s"
                )

            # Save checkpoints
            save_dict = {
                "epoch": epoch + 1,
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
                "best_total_test_loss": best_mean_test_loss,
                "current_total_test_loss": total_test_loss,
            }

            if total_test_loss < best_mean_test_loss:
                best_mean_test_loss = total_test_loss
                save_dict["best_total_test_loss"] = best_mean_test_loss
                best_path = os.path.join(self.save_dir, "best.pth")
                torch.save(save_dict, best_path)
                tqdm.write(f"Saved new best model to {best_path} with total test loss: {best_mean_test_loss:.4f}")

            if (epoch + 1) % self.config.train.save_interval == 0 or epoch == self.config.train.num_epochs - 1:
                epoch_path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pth")
                torch.save(save_dict, epoch_path)
                tqdm.write(f"Saved checkpoint to {epoch_path}")
        
        final_path = os.path.join(self.save_dir, "final.pth")
        torch.save(save_dict, final_path) # save_dict will be from the last epoch
        print(f"Training completed. Final model saved to {final_path}")
        wandb.finish()
        return self.save_dir


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="configs", config_name="train_score", version_base="1.3") # Assuming a 'score.yaml' or similar
    def main(cfg: DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        config = ml_collections.ConfigDict(cfg_dict)
        
        # Ensure timestamp is set if not present, for checkpointing
        if 'timestamp' not in config:
            config.timestamp = time.strftime('%Y%m%d-%H%M%S')
        if 'exp_name' not in config: # Ensure exp_name for checkpointing path
            config.exp_name = config.run_name or "score_experiment"


        print("\n===== Current Score Training Config =====")
        print(OmegaConf.to_yaml(config))
        print("=======================================\n")
        
        trainer = ScoreTrainer(config)
        save_directory = trainer.train()
        print(f"All checkpoints saved in: {save_directory}")

    main()
