import os
import time

import ml_collections
import torch.nn.functional as F  # 确保导入了 F
from sampler import Sampler
import ml_collections
import copy

import wandb
from tqdm import trange, tqdm
import numpy as np
import torch
from utils.protos_utils import (
    compute_protos_from,
)
from models.HVAE import HVAE
from sampler import Sampler
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
from utils.manifolds_utils import (
    get_manifold,
)
from utils.data_utils import MyDataset
from layers.Decoders import Classifier, LogReg  # 使用 LogReg 替代 Classifier
from sklearn.neighbors import KNeighborsClassifier


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        # -------- wandb --------
        mode = "disabled"
        if not self.config.debug:
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
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.run_name = self.config.run_name
        self.dataset = MyDataset(self.config)


    def train_ae(self):
        ts = self.config.timestamp
        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # -------- Load data models, optimizers --------
        (
            self.train_loader,
            self.test_loader,
        ) = self.dataset.get_loaders()
        (
            self.model,
            self.optimizer,
            self.scheduler,
        ) = load_model_optimizer(
            self.config,
            self.config.model.to_dict(),
        )
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.4fM" % (total / 1e6))
        self.encoder=self.model.encoder
        # -------- metrics相关初始化 --------
        self.N_way = self.config.fsl_task.N_way
        self.K_shot = self.config.fsl_task.K_shot
        self.R_query = self.config.fsl_task.R_query
        best_acc = 0.0
        # -------- Training --------
        best_mean_test_loss = 1e10

        t = trange(
            0,
            (self.config.train.num_epochs),
            desc="[Epoch]",
            leave=False,
        )
        for epoch in t:
            self.total_train_loss = []
            self.total_test_loss = []
            self.test_kl_loss = []
            self.test_edge_loss = []
            self.test_rec_loss = []
            self.test_proto_loss = []
            t_start = time.time()
            # train
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, adj, labels = load_batch(
                    batch,
                    self.device,
                )
                (
                    rec_loss,
                    kl_loss,
                    edge_loss,
                    proto_loss,
                ) = self.model(x, adj, labels)
                loss = (
                    rec_loss
                    + self.config.train.kl_regularization * kl_loss
                    + self.config.train.edge_weight * edge_loss
                    + self.config.train.proto_weight * proto_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.grad_norm,
                )
                self.optimizer.step()
                self.total_train_loss.append(loss.item())
            if self.config.train.lr_schedule:
                self.scheduler.step()
            # loss evaluation
            self.model.eval()
            for step, test_batch in enumerate(self.test_loader):
                x, adj, labels = load_batch(
                    test_batch,
                    self.device,
                )
                with torch.no_grad():
                    (
                        rec_loss,
                        kl_loss,
                        edge_loss,
                        proto_loss,
                    ) = self.model(x, adj, labels)
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
            mean_test_proto_loss = np.mean(self.test_proto_loss)



            # -------- Save checkpoints --------
            save_dir = f"./checkpoints/{self.config.data.name}/{self.config.exp_name}/{self.config.timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            # 构建保存信息
            save_dict = {
                "epoch": epoch,
                "model_config": self.config.to_dict(),
                "ae_state_dict": self.model.state_dict(),
                "best_loss": best_mean_test_loss,
                "current_loss": mean_total_test_loss,
            }
            # 判断并保存最佳模型
            if mean_total_test_loss < best_mean_test_loss:
                best_mean_test_loss = mean_total_test_loss
                save_dict["best_loss"] = best_mean_test_loss
                os.makedirs(
                    os.path.dirname(f"{save_dir}/best.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/best.pth",
                )
            wandb.log(
                {
                    "epoch": epoch,
                    "total_test_loss": mean_total_test_loss,
                    "total_train_loss": mean_total_train_loss,
                    "test_edge_loss": mean_test_edge_loss,
                    "test_kl_loss": mean_test_kl_loss,
                    "test_rec_loss": mean_test_rec_loss,
                    "test_proto_loss": mean_test_proto_loss,
                },
                commit=True,
            )

            # -------- Metric评估 --------
            best_acc= 0.0  # Default if not evaluated in this epoch

            eval_interval = getattr(self.config.train, 'eval_interval', None)
            do_eval = False
            if eval_interval is not None:
                try:
                    do_eval = ((epoch + 1) % eval_interval == 0)
                except Exception:
                    do_eval = False
            # 如果没有设置 eval_interval，只在最后一个 epoch 评估
            if do_eval or (epoch == self.config.train.num_epochs - 1):
                mean_acc, std_acc, best_acc = self.meta_eval(
                    epoch, f"{save_dir}/classifier", is_train=False
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "meta_test_mean_fsl_acc": mean_acc,
                        "meta_test_std_fsl_acc": std_acc,
                    },
                    commit=True,
                )
                mean_acc, std_acc, best_acc = self.meta_eval(
                    epoch, f"{save_dir}/classifier", is_train=True
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "meta_train_mean_fsl_acc": mean_acc,
                        "meta_train_std_fsl_acc": std_acc,
                    },
                    commit=True,
                )

            # -------- Metric评估结束 --------

        return self.run_name

    def train_score(self, ts=None):
        # ts = self.config.timestamp
        if self.config.model.ae_path is None:
            self.encoder = None
            self.manifold = get_manifold(
                self.config.model.manifold,
                self.config.model.c,
            )
        else:
            checkpoint = torch.load(
                self.config.model.ae_path, map_location=self.config.device
            )
            AE_state_dict = checkpoint["ae_state_dict"]
            AE_config = ml_collections.ConfigDict(checkpoint["model_config"])
            ae = HVAE(AE_config)
            ae.load_state_dict(AE_state_dict)
            ae.encoder.requires_grad_(False)
            self.encoder = ae.encoder.to(self.device).eval()

            self.manifold = self.encoder.manifold

        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # -------- Load data models, optimizers, ema --------
        (
            self.train_loader,
            self.test_loader,
        ) = self.dataset.get_loaders()
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
        total = sum(
            [param.nelement() for param in self.model_x.parameters()]
            + [param.nelement() for param in self.model_adj.parameters()]
        )
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.ema_x = load_ema(
            self.model_x,
            decay=self.config.train.ema,
        )
        self.ema_adj = load_ema(
            self.model_adj,
            decay=self.config.train.ema,
        )

        # region compute protos
        # 计算元训练集的 protos

        protos_train = compute_protos_from(
            self.encoder,
            self.train_loader,
            self.device,
        )
        protos_test = compute_protos_from(
            self.encoder,
            self.test_loader,
            self.device,
        )

        # end region

        self.loss_fn = load_loss_fn(
            self.config,
            self.manifold,
            encoder=self.encoder,
        )
        # -------- 轮次--------
        best_mean_test_loss = 1e10  # Initialize best mean test loss
        for epoch in trange(
            0,
            (self.config.train.num_epochs),
            desc="[Epoch]",
            position=1,
            leave=False,
        ):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()

            # region train
            for _, train_b in enumerate(self.train_loader):
                x, adj, labels = load_batch(
                    train_b,
                    self.device,
                )

                loss_x, loss_adj = self.loss_fn(
                    self.model_x,
                    self.model_adj,
                    x,
                    adj,
                    labels,
                    protos_train,
                )
                if torch.isnan(loss_x):
                    raise ValueError("NaN")
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                loss_x.backward()
                loss_adj.backward()
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

                x, adj, labels = load_batch(
                    test_b,
                    self.device,
                )
                # Include labels in the loss_subject tuple
                loss_subject = (
                    x,
                    adj,
                    labels,
                )

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    # Now *loss_subject will unpack x, adj, and labels
                    loss_x, loss_adj = self.loss_fn(
                        self.model_x,
                        self.model_adj,
                        *loss_subject,
                        protos_test,
                    )
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)
            total_test_loss = mean_test_x + mean_test_adj  # Calculate total test loss
            # endregion

            # region -------- Log losses --------
            if (
                epoch % self.config.train.print_interval
                == self.config.train.print_interval - 1
            ):
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
                "model_config": self.config.to_dict(),  # 更安全
                "params_x": self.params_x,
                "params_adj": self.params_adj,
                "x_state_dict": self.model_x.state_dict(),
                "adj_state_dict": self.model_adj.state_dict(),
                "ema_x": self.ema_x.state_dict(),
                "ema_adj": self.ema_adj.state_dict(),
                "best_loss": best_mean_test_loss,
                "current_loss": total_test_loss,
            }

            # 按固定间隔保存epoch检查点
            if (
                epoch % self.config.train.save_interval
                == self.config.train.save_interval - 1
            ):
                os.makedirs(
                    os.path.dirname(f"{save_dir}/epoch_{epoch}.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    os.path.join(save_dir, f"epoch_{epoch}.pth"),
                )

            # 判断并保存最佳模型
            if total_test_loss < best_mean_test_loss:
                best_mean_test_loss = total_test_loss
                best_dict = save_dict.copy()
                best_dict["best_loss"] = best_mean_test_loss
                os.makedirs(
                    os.path.dirname(f"{save_dir}/best.pth"),
                    exist_ok=True,
                )
                torch.save(
                    best_dict,
                    os.path.join(save_dir, "best.pth"),
                )

            # 保存最终模型
            if epoch == self.config.train.num_epochs - 1:
                os.makedirs(
                    os.path.dirname(f"{save_dir}/final.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    os.path.join(save_dir, f"final.pth"),
                )
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
            # endregion

        print(" ")
        return self.run_name

    def train_fsl(self):
        """
        Train the few-shot learning model using the configuration from fsl.yaml
        """
        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # Load pretrained encoder
        if self.config.ae_path is None:
            raise ValueError("No autoencoder path specified in config")

        checkpoint = torch.load(
            self.config.ae_path, map_location=self.device, weights_only=False
        )
        AE_state_dict = checkpoint["ae_state_dict"]
        AE_config = ml_collections.ConfigDict(checkpoint["model_config"])
        self.config.model = AE_config.model
        ae = HVAE(AE_config)
        ae.load_state_dict(AE_state_dict, strict=False)
        ae.encoder.requires_grad_(False)
        self.encoder = ae.encoder.to(self.device).eval()


        # 初始化必要的参数
        self.N_way = self.config.fsl_task.N_way
        self.K_shot = self.config.fsl_task.K_shot
        self.R_query = self.config.fsl_task.R_query



        # 传入当前轮次(0)和初始最佳准确率(0.0)
        save_dir = f"./checkpoints/{self.config.data.name}/{self.config.exp_name}/{self.config.timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        mean_acc, std_acc, best_acc = self.meta_eval(epoch=0, save_dir=f'{save_dir}/classifier', is_train=True)
        print(f"Meta-test results: Mean Accuracy: {mean_acc:.4f}, Std Accuracy: {std_acc:.4f}, Best Accuracy: {best_acc:.4f}")
        return self.config.run_name

    def train_one_step(
            self,
            is_train,
            model,
            dataset,
            device,
            N_way,
            K_shot,
            R_query,
            log_model,
            opt,
            num_epochs,
            save_dir,
            config_obj,
            test_start_idx=None,
        ):
            model.eval()

            # 采样一个 few-shot 任务
            sample_kwargs = dict(
                is_train=is_train,
                N_way=N_way,
                K_shot=K_shot,
                R_query=R_query,
            )
            if (not is_train) and (test_start_idx is not None):
                sample_kwargs['test_start_idx'] = test_start_idx
            current_task = dataset.sample_one_task(**sample_kwargs)

            # 支持集
            support_x = current_task["support_set"]["x"].to(device)
            support_adj = current_task["support_set"]["adj"].to(device)
            support_label = current_task["support_set"]["label"].to(device)
            support_dataset_obj = torch.utils.data.TensorDataset(support_x, support_adj, support_label)
            support_loader = torch.utils.data.DataLoader(support_dataset_obj, batch_size=len(support_x), shuffle=False)

            # 可选：采样器增强
            if config_obj.fsl_task.get("use_sampler_augmentation", False):
                from sampler import Sampler
                import copy
                sampler_config = copy.deepcopy(config_obj)
                sampler_config.dataloader = support_loader
                sampler = Sampler(sampler_config)
                augmented_support_loader = sampler.sample(need_eval=False)
                augmented_x = augmented_support_loader.dataset.tensors[0].to(device)
                augmented_adj = augmented_support_loader.dataset.tensors[1].to(device)
                augmented_label = augmented_support_loader.dataset.tensors[2].to(device)
                if augmented_label.max() >= N_way:
                    augmented_label = torch.clamp(augmented_label, 0, N_way - 1)
                combined_dataset = torch.utils.data.TensorDataset(augmented_x, augmented_adj, augmented_label)
                support_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=len(support_x), shuffle=True)

            # 训练线性探针
            log_model.train()
            best_loss = float('inf')
            wait = 0
            patience = 20
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, f"{config_obj.data.name}_lr.pkl")

            for batch_x, batch_adj, batch_label in support_loader:
                node_masks = torch.stack([node_flags(adj) for adj in batch_adj])
                with torch.no_grad():
                    posterior = model(batch_x, batch_adj, node_masks)
                    graph_embs = posterior.mode()
                    if graph_embs.dim() == 3 and graph_embs.size(1) == 1:
                        graph_embs = graph_embs.squeeze(1)
                    # 先将所有节点嵌入投影回欧氏空间，再做mean池化
                    if self.encoder.manifold is not None:
                        graph_embs = self.encoder.manifold.logmap0(graph_embs)
                    batch_embeddings = graph_embs.mean(dim=1)

                for _ in range(num_epochs):
                    opt.zero_grad()
                    logits = log_model(batch_embeddings)
                    loss = torch.nn.functional.cross_entropy(logits, batch_label.long())
                    l2_reg = torch.tensor(0.0).to(device)
                    for param in log_model.parameters():
                        l2_reg += torch.norm(param)
                    loss = loss + 0.1 * l2_reg
                    loss.backward()
                    opt.step()
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        wait = 0
                        torch.save(log_model.state_dict(), best_model_path)
                    else:
                        wait += 1
                    if wait > patience:
                        break

            if os.path.exists(best_model_path):
                log_model.load_state_dict(torch.load(best_model_path, weights_only=True))

            log_model.eval()

            # 查询集
            query_x = current_task["query_set"]["x"].to(device)
            query_adj = current_task["query_set"]["adj"].to(device)
            query_label = current_task["query_set"]["label"].to(device)
            query_len = query_label.shape[0]
            effective_len = query_len
            if current_task["append_count"] != 0:
                effective_len = query_len - current_task["append_count"]
                query_x = query_x[:effective_len]
                query_adj = query_adj[:effective_len]
                query_label = query_label[:effective_len]
            query_dataset_obj = torch.utils.data.TensorDataset(query_x, query_adj, query_label)
            query_loader = torch.utils.data.DataLoader(query_dataset_obj, batch_size=effective_len, shuffle=False)

            query_data = []
            for batch_x, batch_adj, batch_label in query_loader:
                node_masks = torch.stack([node_flags(adj) for adj in batch_adj])
                with torch.no_grad():
                    posterior = model(batch_x, batch_adj, node_masks)
                    graph_embs = posterior.mode()
                    if graph_embs.dim() == 3 and graph_embs.size(1) == 1:
                        graph_embs = graph_embs.squeeze(1)
                    # 先将所有节点嵌入投影回欧氏空间，再做mean池化
                    if self.encoder.manifold is not None:
                        graph_embs = self.encoder.manifold.logmap0(graph_embs)
                    batch_embeddings = graph_embs.mean(dim=1)
                query_data.append(batch_embeddings)
            if len(query_data) == 0:
                raise RuntimeError("No query data was collected: query_loader is empty or query set is empty.")
            query_data = torch.cat(query_data, dim=0)
            query_labels = query_label[:effective_len]
            logits = log_model(query_data)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == query_labels).float() / query_labels.shape[0]
            test_acc = acc.cpu().numpy()
            return test_acc

    def meta_eval(self, epoch, save_dir, is_train=False):

        # Initialize classifier and optimizer for FSL evaluation for this epoch
        fsl_log_model = LogReg(self.config.model.dim , self.N_way).to(self.device)
        fsl_opt = torch.optim.Adam(fsl_log_model.parameters(), lr=self.config.train.lr)

        current_epoch_fsl_accs = []
        start_test_idx = 0

        # 选择数据源
        if is_train:
            total_graphs = len(self.dataset.train_graphs)
            total_classes = self.dataset.train_classes_num
        else:
            total_graphs = len(self.dataset.test_graphs)
            total_classes = self.dataset.test_classes_num

        total_tasks = (total_graphs - self.K_shot * total_classes) // (
            self.N_way * self.R_query
        )
        if total_tasks > 0:
            task_progress_desc = f"[Epoch {epoch} FSL Eval]"
            tasks_done_count = 0
            pbar = tqdm(total=total_tasks, desc=task_progress_desc, leave=False, dynamic_ncols=True)
            while (
                start_test_idx
                < total_graphs - self.K_shot * total_classes
            ) and (tasks_done_count < total_tasks):
                task_acc = self.train_one_step(
                    is_train=is_train,
                    model=self.encoder,
                    dataset=self.dataset,
                    device=self.device,
                    N_way=self.N_way,
                    K_shot=self.K_shot,
                    R_query=self.R_query,
                    log_model=fsl_log_model,
                    opt=fsl_opt,
                    num_epochs=self.config.fsl_task.num_epochs,
                    save_dir=save_dir,
                    config_obj=self.config,
                    test_start_idx=start_test_idx,
                )
                current_epoch_fsl_accs.append(task_acc)
                start_test_idx += self.N_way * self.R_query
                tasks_done_count += 1
                pbar.update(1)
            pbar.close()
        elif total_graphs > 0 and total_tasks == 0:
            print(
                f"Warning: Not enough data for a full FSL evaluation task batch in epoch {epoch} (total_tasks = {total_tasks})."
            )

        mean_fsl_acc_epoch = np.mean(current_epoch_fsl_accs)
        std_fsl_acc_epoch = np.std(current_epoch_fsl_accs)
        best_fsl_acc_epoch = np.max(current_epoch_fsl_accs)

        return mean_fsl_acc_epoch, std_fsl_acc_epoch, best_fsl_acc_epoch
