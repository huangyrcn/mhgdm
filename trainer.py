import os
import time


import ml_collections

import wandb
from tqdm import  trange
import numpy as np
import torch
from utils.protos_utils import compute_protos_from
from models.HVAE import HVAE
from sampler import  Sampler
from utils.graph_utils import node_flags
from utils.loader import (
    load_seed,
    load_device,
    load_data,
    load_model_optimizer,
    load_ema,
    load_loss_fn,
    load_batch,
)
from utils.logger import Logger, set_log, start_log, train_log
from utils.manifolds_utils import get_manifold


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.run_name =self.config.run_name



    def train_ae(self):
        ts = self.config.timestamp

        mode = "disabled"
        if not self.config.debug: mode = "online" if self.config.wandb.online else "offline"
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.run_name,
            config=self.config.to_dict(),
            settings=wandb.Settings(_disable_stats=True),
            mode=mode,
            dir=os.path.join("logs", "wandb")  # 👈 将 wandb 日志写入 logs/wandb 目录
        )

        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # -------- Load data models, optimizers --------
        self.train_loader, self.test_loader = load_data(self.config)
        self.model, self.optimizer, self.scheduler = load_model_optimizer(
            self.config, self.config.model.to_dict()
        )
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.4fM" % (total / 1e6))

        logger = Logger(str(os.path.join(self.log_dir, f"{self.run_name}.log")), mode="a")
        logger.log(f"{self.run_name}", verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        # -------- Training --------

        best_mean_test_loss = 1e10

        for epoch in trange(0, (self.config.train.num_epochs), desc="[Epoch]", leave=False):

            self.total_train_loss = []
            self.total_test_loss = []
            self.test_kl_loss = []
            self.test_edge_loss = []
            self.test_rec_loss = []
            self.test_proto_loss = []  # 新增：记录测试时的原型损失
            t_start = time.time()

            # train

            self.model.train()

            for step, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, adj, labels = load_batch(batch, self.device)
                rec_loss, kl_loss, edge_loss, proto_loss = self.model(x, adj, labels)

                loss = (
                    rec_loss
                    + self.config.train.kl_regularization * kl_loss
                    + self.config.train.edge_weight * edge_loss
                    + self.config.train.proto_weight * proto_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                self.optimizer.step()
                self.total_train_loss.append(loss.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            # loss evaluation
            self.model.eval()
            for _, test_batch in enumerate(self.test_loader):
                x, adj, labels = load_batch(test_batch, self.device)
                with torch.no_grad():
                    rec_loss, kl_loss, edge_loss, proto_loss = self.model(x, adj, labels)
                    loss = (
                        rec_loss
                        + self.config.train.kl_regularization * kl_loss
                        + self.config.train.edge_weight * edge_loss
                        + self.config.train.proto_weight * proto_loss
                    )

                    self.total_test_loss.append(loss.item())
                    self.test_rec_loss.append(rec_loss.item())
                    self.test_kl_loss.append(kl_loss.item())
                    self.test_edge_loss.append(edge_loss.item())
                    self.test_proto_loss.append(proto_loss.item())
            mean_total_train_loss = np.mean(self.total_train_loss)
            mean_total_test_loss = np.mean(self.total_test_loss)
            mean_test_rec_loss = np.mean(self.test_rec_loss)
            mean_test_kl_loss = np.mean(self.test_kl_loss)
            mean_test_edge_loss = np.mean(self.test_edge_loss)
            mean_test_proto_loss = np.mean(self.test_proto_loss)  # 计算平均原型损失
            if (
                "HGCN" in [self.config.model.encoder, self.config.model.decoder]
            ) and self.config.model.learnable_c:
                self.model.show_curvatures()

            # encoder evaluation
            # encoder_acc, encoder_nmi, proto_match_acc = evaluate_encoder_metrics(
            #     self.model.encoder, self.model.graph_prototypes, self.test_loader
            # )

            # -------- Save checkpoints --------
            save_dir = f"./checkpoints/{self.config.data.name}/{self.config.exp_name}/{self.config.timestamp}"
            os.makedirs(save_dir, exist_ok=True)

            # 构建保存信息
            save_dict = {
                "epoch": epoch,
                "model_config": self.config.to_dict(),
                "ae_state_dict": self.model.state_dict(),
                "best_loss": best_mean_test_loss,
                "current_loss": mean_total_test_loss
            }

            # 按固定间隔保存epoch检查点
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
                os.makedirs(os.path.dirname(f"{save_dir}/epoch_{epoch}.pth"), exist_ok=True)
                torch.save(save_dict, f"{save_dir}/epoch_{epoch}.pth")

            # 判断并保存最佳模型
            if mean_total_test_loss < best_mean_test_loss:
                best_mean_test_loss = mean_total_test_loss
                save_dict["best_loss"] = best_mean_test_loss
                os.makedirs(os.path.dirname(f"{save_dir}/best.pth"), exist_ok=True)
                torch.save(save_dict, f"{save_dir}/best.pth")

            # 保存最终模型
            if epoch == self.config.train.num_epochs - 1:
                os.makedirs(os.path.dirname(f"{save_dir}/final.pth"), exist_ok=True)
                torch.save(save_dict, f"{save_dir}/final.pth")

            wandb.log(
                {
                    "epoch": epoch,
                    "total_test_loss": mean_total_test_loss,
                    "total_train_loss": mean_total_train_loss,
                    "test_edge_loss": mean_test_edge_loss,
                    "test_kl_loss": mean_test_kl_loss,
                    "test_rec_loss": mean_test_rec_loss,
                    "test_proto_loss": mean_test_proto_loss,  # 添加到wandb日志
                },
                commit=True,
            )

            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                logger.log(
                    f"{epoch + 1:03d} | {time.time() - t_start:.2f}s | "
                    f"total train loss: {mean_total_train_loss:.3e} | "
                    f"total test loss: {mean_total_test_loss:.3e} | "
                    f"test rec loss: {mean_test_rec_loss:.3e} | "
                    f"test kl loss: {mean_test_kl_loss:.3e} | "
                    f"test edge loss: {mean_test_edge_loss:.3e} | "
                    f"test proto loss: {mean_test_proto_loss:.3e} |",  # 添加到logger日志
                    verbose=False,
                )

        print(" ")

        return self.run_name

    def train_score(self, ts=None):
        # ts = self.config.timestamp
        if self.config.model.ae_path is None:
            Encoder = None
            self.manifold = get_manifold(self.config.model.manifold, self.config.model.c)
        else:
            checkpoint = torch.load(self.config.model.ae_path, map_location=self.config.device,weights_only=False)
            AE_state_dict = checkpoint["ae_state_dict"]
            AE_config = ml_collections.ConfigDict(checkpoint["model_config"])
            AE_config.model.dropout = 0
            ae = HVAE(AE_config)
            ae.load_state_dict(AE_state_dict, strict=False)
            for name, param in ae.named_parameters():
                if "encoder" in name or "decoder" in name:
                    param.requires_grad = False
            Encoder = ae.encoder.to(self.device)
            self.manifold = Encoder.manifold


        # -------- wandb --------
        mode = "disabled"
        if not self.config.debug: mode = "online" if self.config.wandb.online else "offline"
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.run_name,
            config=self.config.to_dict(),
            settings=wandb.Settings(_disable_stats=True),
            mode=mode,
            dir=os.path.join("logs", "wandb")  # 👈 将 wandb 日志写入 logs/wandb 目录
        )

        
        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # -------- Load data models, optimizers, ema --------
        self.train_loader, self.test_loader= load_data(self.config)
        self.params_x = self.config.model.x.to_dict()
        self.params_x['manifold'] = self.manifold
        self.params_adj =self.config.model.adj.to_dict()
        self.params_adj['manifold'] = self.manifold
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(
            self.config,self.params_x, 
        )
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(
            self.config,self.params_adj)
        total = sum(
            [param.nelement() for param in self.model_x.parameters()]
            + [param.nelement() for param in self.model_adj.parameters()]
        )
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f"{self.run_name}.log")), mode="a")
        logger.log(f"{self.run_name}", verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)


        # region compute protos
        # 计算元训练集的 protos

        protos_train = compute_protos_from(Encoder, self.train_loader, self.device)
        protos_test = compute_protos_from(Encoder, self.test_loader, self.device)

        # end region
        
        self.loss_fn = load_loss_fn(self.config, self.manifold, encoder=Encoder)
        # -------- 轮次--------
        best_mean_test_loss = 1e10  # Initialize best mean test loss
        for epoch in trange(            0, (self.config.train.num_epochs), desc="[Epoch]", position=1, leave=False ):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()
        

            # region train
            for _, train_b in enumerate(self.train_loader):
                x, adj, labels= load_batch(
                    train_b, self.device
                ) 
                
                loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, x, adj,labels,protos_train)
                if torch.isnan(loss_x):
                    raise ValueError("NaN")
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                loss_x.backward()
                loss_adj.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_x.parameters(), self.config.train.grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model_adj.parameters(), self.config.train.grad_norm
                )

                self.optimizer_x.step()
                self.optimizer_adj.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()
            # endregion
            
            # region test
            self.model_x.eval()
            self.model_adj.eval()
                   

            for _, test_b in enumerate(self.test_loader):

                x, adj, labels = load_batch(test_b, self.device)
                # Include labels in the loss_subject tuple
                loss_subject = (x, adj, labels) 

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    # Now *loss_subject will unpack x, adj, and labels
                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject,protos_test) 
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)
            total_test_loss = mean_test_x + mean_test_adj # Calculate total test loss
            # endregion
            
            # region -------- Log losses --------
            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                logger.log(
                    f"{epoch+1:03d} | {time.time()-t_start:.2f}s | "
                    f"test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | "
                    f"train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ",
                    verbose=False,
                )
                wandb.log(
                    {   
                        "epoch": epoch,
                        "Test x": mean_test_x,
                        "test adj": mean_test_adj,
                        "train x": mean_train_x,
                        "train adj": mean_train_adj,
                        "epoch": epoch + 1,
                    },
                    commit=True,
                )
            # endregion
            
            # region -------- Save checkpoints --------
            save_dir = f"./checkpoints/{self.config.data.name}/{self.config.exp_name}/{self.config.timestamp}"
            os.makedirs(save_dir, exist_ok=True)

            # 构建保存信息
            save_dict = {
                "epoch": epoch,
                "model_config": self.config,
                "params_x": self.params_x,
                "params_adj": self.params_adj,
                "x_state_dict": self.model_x.state_dict(),
                "adj_state_dict": self.model_adj.state_dict(),
                "ema_x": self.ema_x.state_dict(),
                "ema_adj": self.ema_adj.state_dict(),
                "best_loss": best_mean_test_loss,
                "current_loss": total_test_loss
            }

            # 按固定间隔保存epoch检查点
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
                os.makedirs(os.path.dirname(f"{save_dir}/epoch_{epoch}.pth"), exist_ok=True)
                torch.save(save_dict, f"{save_dir}/epoch_{epoch}.pth")

            # 判断并保存最佳模型
            if total_test_loss < best_mean_test_loss:
                best_mean_test_loss = total_test_loss
                save_dict["best_loss"] = best_mean_test_loss
                os.makedirs(os.path.dirname(f"{save_dir}/best.pth"), exist_ok=True)
                torch.save(save_dict, f"{save_dir}/best.pth")

            # 保存最终模型
            if epoch == self.config.train.num_epochs - 1:
                os.makedirs(os.path.dirname(f"{save_dir}/final.pth"), exist_ok=True)
                torch.save(save_dict, f"{save_dir}/final.pth")
            # endregion

            # region -------- Sample evaluation --------
            # if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
            #     self.config.sampler.snr_x = "0.1" 
            #     self.config.sampler.scale_eps_x = "1.0" # Corrected typo and type
            #     self.config.sampler.ckp_path = f"{save_dir}/epoch_{epoch}.pth"
            #     if self.config.data.name == "ENZYMES":
            #         eval_dict = Sampler(self.config).sample(independent=False)    
            #     eval_dict["epoch"] = epoch + 1
            #     wandb.log(eval_dict, commit=True)
            #     logger.log(f"[EPOCH {epoch + 1:04d}] Saved! \n" + str(eval_dict), verbose=False)
            # endregion
            
            # endreigon
            # region -------- Print losses --------
            # if epoch % self.config.train.print_interval == self.confiffg.train.print_interval - 1:
            #     tqdm.write(
            #         f"[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | "
            #         f"test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}"
            #     )
            #endregion
        
        print(" ")
        return self.run_name

    def train_fsl(self):#元测试模型
        # 从dataset中采样元测试任务
        # 使用元测试任务的支持集编码计算原型
        # 然后基于原型引导生成扩充支持集
        # 然后使用支持集微调分类任务模型
        # 并最终在查询集上评估分类性能
        print("Meta-testing...")

        # -------- Load Pre-trained Models (Score Network and optional AE) --------
        if self.config.model.ae_path is None:
            Encoder = None
            self.manifold = get_manifold(self.config.model.manifold, self.config.model.c)
        else:
            checkpoint_ae = torch.load(self.config.model.ae_path, map_location=self.config.device, weights_only=False)
            AE_config = ml_collections.ConfigDict(checkpoint_ae["model_config"])
            AE_config.model.dropout = 0 # Ensure dropout is off during inference/testing
            ae = HVAE(AE_config)
            ae.load_state_dict(checkpoint_ae["ae_state_dict"], strict=False)
            ae.eval() # Set AE to evaluation mode
            for param in ae.parameters():
                param.requires_grad = False # Freeze AE parameters
            Encoder = ae.encoder.to(self.device)
            self.manifold = Encoder.manifold
            print(f"Loaded AE from: {self.config.model.ae_path}")


        # -------- Load FSL Data --------
        # Assuming load_data can handle FSL task setup or a new loader is needed
        # For now, let's assume test_loader provides meta-test tasks
        # Each batch in test_loader could represent one task (support + query)
        # Or, we might need a dedicated FSL dataloader.
        # Let's proceed assuming test_loader yields tasks.
        _, self.meta_test_loader = load_data(self.config, fsl_task=True) # Modify load_data or use a specific FSL loader

        # -------- Meta-Testing Loop --------
        all_task_accuracies = []
        num_tasks = len(self.meta_test_loader) # Or specify number of tasks in config

        for task_idx, task_data in enumerate(self.meta_test_loader):
            print(f"--- Processing Meta-Test Task {task_idx + 1}/{num_tasks} ---")

            # 1. Split task data into support and query sets
            # This depends heavily on how the FSL dataloader structures the data.
            # Example structure: task_data = (support_x, support_adj, support_y, query_x, query_adj, query_y)
            # Adjust based on your actual FSL data loading implementation.
            # For demonstration, let's assume a function `split_support_query` exists.
            # support_x, support_adj, support_y, query_x, query_adj, query_y = split_support_query(task_data, self.config.fsl.n_way, self.config.fsl.k_shot, self.config.fsl.k_query, self.device)
            
            # Placeholder: Assuming task_data is already structured or needs specific handling
            # Example: Directly load support and query data if loader provides it separately per iteration
            support_x, support_adj, support_labels = load_batch(task_data['support'], self.device) # Adjust based on actual loader output
            query_x, query_adj, query_labels = load_batch(task_data['query'], self.device)       # Adjust based on actual loader output
            
            print(f"Support set size: {support_x.shape[0]}, Query set size: {query_x.shape[0]}")

            # 2. Compute prototypes from the support set using the Encoder
            if Encoder is not None:
                with torch.no_grad():
                    support_z, _ = Encoder(support_x, support_adj) # Get latent representations
                    # Compute prototypes (e.g., mean of embeddings per class)
                    # This requires knowing the class labels (`support_labels`)
                    # Example:
                    # unique_labels = torch.unique(support_labels)
                    # support_protos = torch.stack([support_z[support_labels == label].mean(dim=0) for label in unique_labels])
                    # print(f"Computed {len(unique_labels)} prototypes from support set.")
                    
                    # Placeholder for actual prototype computation logic
                    support_protos = compute_protos_from(Encoder, [(support_x, support_adj, support_labels)], self.device) # Adapt compute_protos_from if needed
                    print(f"Computed prototypes from support set. Shape: {support_protos.shape}")

            else:
                print("Warning: No AE Encoder provided, cannot compute latent prototypes.")
                support_protos = None # Or handle differently if no AE is used

            # 3. Generate augmented support set based on prototypes (Optional)
            # This step involves using the loaded Score Network (model_x, model_adj)
            # and the computed prototypes to guide the generation process.
            # You would need to adapt the Sampler class or create a specific generation function.
            # Example call (conceptual):
            # sampler = Sampler(self.config, self.model_x, self.model_adj, self.manifold, Encoder) # Pass necessary models
            # augmented_x, augmented_adj, augmented_labels = sampler.sample_conditional(
            #     num_samples_per_class=self.config.fsl.augmentation_samples,
            #     prototypes=support_protos,
            #     class_labels=unique_labels # Need the labels corresponding to protos
            # )
            # print(f"Generated {augmented_x.shape[0]} augmented samples.")

            # Combine original support set with augmented data
            # combined_support_x = torch.cat([support_x, augmented_x], dim=0)
            # combined_support_adj = torch.cat([support_adj, augmented_adj], dim=0) # Check adjacency matrix combination logic
            # combined_support_labels = torch.cat([support_labels, augmented_labels], dim=0)
            # print(f"Combined support set size: {combined_support_x.shape[0]}")
            
            # For now, skipping augmentation and using original support set for fine-tuning
            combined_support_x, combined_support_adj, combined_support_labels = support_x, support_adj, support_labels


            # 4. Fine-tune/Train a classifier on the (augmented) support set
            # Define a simple classifier (e.g., Logistic Regression, MLP, or prototype-based)
            # Train this classifier using `combined_support_x/adj/labels` (or their latent embeddings `support_z`)
            
            # Example: Using latent embeddings `support_z` and a simple classifier
            if Encoder is not None:
                with torch.no_grad():
                     combined_support_z, _ = Encoder(combined_support_x, combined_support_adj)
                     query_z, _ = Encoder(query_x, query_adj)

                # Define classifier (e.g., Logistic Regression on latent space)
                classifier = torch.nn.Linear(combined_support_z.size(-1), self.config.fsl.n_way).to(self.device) # n_way classification
                classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.fsl.classifier_lr)
                loss_fn_classifier = torch.nn.CrossEntropyLoss()

                print("Fine-tuning classifier...")
                classifier.train()
                for ft_epoch in range(self.config.fsl.classifier_epochs):
                    classifier_optimizer.zero_grad()
                    logits = classifier(combined_support_z)
                    # Need to map original labels to 0..N-1 for CrossEntropyLoss
                    # Assuming labels are already in this range or a mapping is applied
                    loss = loss_fn_classifier(logits, combined_support_labels)
                    loss.backward()
                    classifier_optimizer.step()
                    if (ft_epoch + 1) % 10 == 0: # Print progress occasionally
                         print(f"  Classifier Fine-tune Epoch {ft_epoch+1}, Loss: {loss.item():.4f}")

                # 5. Evaluate the classifier on the query set
                print("Evaluating on query set...")
                classifier.eval()
                with torch.no_grad():
                    query_logits = classifier(query_z)
                    predictions = torch.argmax(query_logits, dim=1)
                    # Map query_labels similarly if needed
                    correct = (predictions == query_labels).sum().item()
                    accuracy = correct / len(query_labels)
                    all_task_accuracies.append(accuracy)
                    print(f"Task {task_idx + 1} Accuracy: {accuracy:.4f}")

            else:
                 print("Skipping classifier training/evaluation as no Encoder is available.")
                 # Handle evaluation differently if working directly in graph space


        # -------- Final Results --------
        if all_task_accuracies:
            mean_accuracy = np.mean(all_task_accuracies)
            std_accuracy = np.std(all_task_accuracies)
            confidence_interval = 1.96 * std_accuracy / np.sqrt(len(all_task_accuracies)) # 95% CI

            print("\n--- Meta-Testing Summary ---")
            print(f"Number of tasks evaluated: {len(all_task_accuracies)}")
            print(f"Mean Accuracy: {mean_accuracy:.4f}")
            print(f"Standard Deviation: {std_accuracy:.4f}")
            print(f"95% Confidence Interval: +/- {confidence_interval:.4f}")

            # Log results (e.g., to wandb or logger)
            if not self.config.wandb.no_wandb:
                 wandb.log({ 
                     "meta_test_mean_accuracy": mean_accuracy,
                     "meta_test_std_accuracy": std_accuracy,
                     "meta_test_confidence_interval": confidence_interval
                 })
        else:
            print("\n--- Meta-Testing Summary ---")
            print("No tasks were successfully evaluated.")

        print("Meta-testing finished.")
        # Potentially return results or save them
        return {"mean_accuracy": mean_accuracy, "std_accuracy": std_accuracy} if all_task_accuracies else {}

