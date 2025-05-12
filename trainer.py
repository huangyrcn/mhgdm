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
        self.query_size = self.config.fsl_task.query_size
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
            if (
                "HGCN"
                in [
                    self.config.model.encoder,
                    self.config.model.decoder,
                ]
            ) and self.config.model.learnable_c:
                self.model.show_curvatures()



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
            if (epoch + 1) % self.config.train.eval_interval == 0 or (epoch == self.config.train.num_epochs - 1):
                mean_acc   , std_acc, best_acc = self.fsl_test(
                    epoch, f"{save_dir}/classifier"
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "mean_fsl_acc": mean_acc,
                        "std_fsl_acc": std_acc,
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
        self.query_size = self.config.fsl_task.query_size



        # 传入当前轮次(0)和初始最佳准确率(0.0)
        mean_acc, std_acc, best_acc = self.fsl_test(epoch=0)
        # 记录当前轮次的准确率
        wandb.log(
            {
                "epoch": 0,
                "mean_fsl_acc": mean_acc,
                "std_fsl_acc": std_acc,
                "best_fsl_acc": best_acc,
            },
            commit=True,
        )
        return self.config.run_name

    def train_one_step(
        self,
        epoch,
        test_idx,
        model,
        dataset,
        device,
        N_way,
        K_shot,
        query_size,
        log_model,
        opt,
        num_epochs,
        save_dir,
        config_obj
    ):
        """
        Train or evaluate on a single task.

        :param epoch: Current epoch number
        :param test_idx: Index of the test task
        :param model: Encoder model
        :param dataset: Dataset for sampling tasks
        :param device: Computing device
        :param N_way: Number of ways
        :param K_shot: Number of shots
        :param query_size: Size of query set
        :param log_model: Classifier model (e.g., LogReg)
        :param opt: Optimizer for the classifier model
        :param config_obj: Configuration object
        :return: Accuracy for the task
        """
        model.eval()

        # Sample one task using dataset's method
        first_N_class_sample = np.array(list(range(dataset.test_classes_num)))
        current_task = dataset.sample_one_task(
            dataset.test_tasks,
            first_N_class_sample,
            K_shot=K_shot,
            query_size=query_size,
            test_start_idx=test_idx,
        )

        # Get support set data
        support_x = current_task["support_set"]["x"].to(device)
        support_adj = current_task["support_set"]["adj"].to(device)
        support_label = current_task["support_set"]["label"].to(device)

        # 创建支持集的数据加载器
        support_dataset_obj = torch.utils.data.TensorDataset(
            support_x, support_adj, support_label
        )  # Renamed
        support_loader = torch.utils.data.DataLoader(
            support_dataset_obj, batch_size=len(support_x), shuffle=False
        )

        # 检查是否启用采样器增强
        if config_obj.fsl_task.get("use_sampler_augmentation", False):  # 默认为 False
            # 使用 sampler 扩充 support_loader
            # 创建一个临时配置对象，用于初始化 Sampler
            from sampler import Sampler
            import ml_collections
            import copy

            # 创建采样器配置
            sampler_config = copy.deepcopy(config_obj)

            # 设置采样器要使用的 dataloader
            sampler_config.dataloader = support_loader

            sampler = Sampler(sampler_config)
            augmented_support_loader = sampler.sample(need_eval=False)

            # 将原始支持集与增强数据结合，而不是替换
            # 从两个DataLoader中提取数据
            augmented_x = augmented_support_loader.dataset.tensors[0]
            augmented_adj = augmented_support_loader.dataset.tensors[1]
            augmented_label = augmented_support_loader.dataset.tensors[2]

            # 确保所有张量都在同一个设备上 (device)
            augmented_x = augmented_x.to(device)
            augmented_adj = augmented_adj.to(device)
            augmented_label = augmented_label.to(device)

            # 确保标签在有效范围内 (0 到 N_way-1)
            if augmented_label.max() >= N_way:
                augmented_label = torch.clamp(augmented_label, 0, N_way - 1)

            # 创建组合数据集
            combined_dataset = torch.utils.data.TensorDataset(
                augmented_x, augmented_adj, augmented_label
            )

            # 创建新的DataLoader
            combined_loader = torch.utils.data.DataLoader(
                combined_dataset,
                batch_size=len(support_x),  # 使用原始支持集的大小作为批次大小
                shuffle=True,  # 打乱数据顺序
            )

            # 使用组合后的数据加载器
            support_loader = combined_loader


        log_model.train()
        best_loss = float('inf')
        wait = 0
        patience = 20  # 可根据需要调整
        
        # 使用传入的 save_dir 作为保存目录
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f"{config_obj.data.name}_lr.pkl")

        for batch_x, batch_adj, batch_label in support_loader:
            node_masks = torch.stack([node_flags(adj) for adj in batch_adj])
            with torch.no_grad():
                posterior = model(batch_x, batch_adj, node_masks)
                graph_embs = posterior.mode()
                if graph_embs.dim() == 3 and graph_embs.size(1) == 1:
                    graph_embs = graph_embs.squeeze(1)
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

                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    wait = 0
                    torch.save(log_model.state_dict(), best_model_path) # 使用新的路径
                else:
                    wait += 1
                if wait > patience:

                    break

        # 恢复最优模型参数
        # 确保在加载前检查文件是否存在，以增加稳健性
        if os.path.exists(best_model_path):
            log_model.load_state_dict(torch.load(best_model_path)) # 使用新的路径

        # 评估阶段
        log_model.eval()

        # Get query set data
        query_x = current_task["query_set"]["x"].to(device)
        query_adj = current_task["query_set"]["adj"].to(device)
        query_label = current_task["query_set"]["label"].to(device)

        # 创建查询集的数据加载器
        query_dataset_obj = torch.utils.data.TensorDataset(
            query_x, query_adj, query_label
        )  # Renamed

        # 处理可能的append_count
        query_len = query_label.shape[0]
        effective_len = query_len
        if current_task["append_count"] != 0:
            effective_len = query_len - current_task["append_count"]
            query_dataset_obj = torch.utils.data.TensorDataset(  # Renamed
                query_x[:effective_len],
                query_adj[:effective_len],
                query_label[:effective_len],
            )

        query_loader = torch.utils.data.DataLoader(
            query_dataset_obj, batch_size=effective_len, shuffle=False  # Renamed
        )

        # Process query set as batches
        query_data = []
        for batch_x, batch_adj, batch_label in query_loader:
            # 为整个批次计算 node_mask
            node_masks = torch.stack([node_flags(adj) for adj in batch_adj])

            with torch.no_grad():
                # 直接处理整个批次
                posterior = model(batch_x, batch_adj, node_masks)
                graph_embs = posterior.mode()

                # 处理维度
                if graph_embs.dim() == 3 and graph_embs.size(1) == 1:
                    graph_embs = graph_embs.squeeze(1)

                # 对每个图的节点嵌入取平均，得到图级嵌入
                graph_embs = graph_embs.mean(dim=1)

                # 添加到结果列表
                query_data.append(graph_embs)

        # Concatenate all batch results
        query_data = torch.cat(query_data, dim=0)
        query_labels = query_label[:effective_len]  # 只使用有效部分的标签
        # Output query set predictions and true labels

        # Calculate accuracy
        logits = log_model(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_labels).float() / query_labels.shape[0]
        test_acc = acc.cpu().numpy()
        return test_acc

    def fsl_test(self, epoch, save_dir):
        """
        Performs Few-Shot Learning (FSL) evaluation for the current epoch.

        Args:
            epoch (int): The current training epoch.

        Returns:
            tuple: (mean_fsl_acc_epoch, std_fsl_acc_epoch, best_fsl_acc_epoch)
                   mean_fsl_acc_epoch (float): Mean FSL accuracy for the current epoch.
                   std_fsl_acc_epoch (float): Std of FSL accuracy for the current epoch.
        """
        # Initialize classifier and optimizer for FSL evaluation for this epoch
        ft_in = self.config.model.dim  # Ensure this uses .dim for correct feature size
        nb_classes = self.N_way
        fsl_log_model = LogReg(ft_in, nb_classes).to(self.device)
        fsl_opt = torch.optim.Adam(fsl_log_model.parameters(), lr=self.config.train.lr)

        current_epoch_fsl_accs = []
        start_test_idx = 0

        # 计算总任务数量，用于进度条显示
        total_test_graphs = len(self.dataset.test_graphs)
        total_test_classes = self.dataset.test_classes_num
        total_tasks = (total_test_graphs - self.K_shot * total_test_classes) // (
            self.N_way * self.query_size
        )
        if total_tasks > 0:
            task_progress_desc = f"[Epoch {epoch} FSL Eval]"
            tasks_done_count = 0
            pbar = tqdm(total=total_tasks, desc=task_progress_desc, leave=False, dynamic_ncols=True)
            while (
                start_test_idx
                < len(self.dataset.test_graphs)
                - self.K_shot * self.dataset.test_classes_num
            ) and (tasks_done_count < total_tasks):
                task_acc = self.train_one_step(
                    epoch=epoch,
                    test_idx=start_test_idx,
                    model=self.encoder,
                    dataset=self.dataset,
                    device=self.device,
                    N_way=self.N_way,
                    K_shot=self.K_shot,
                    query_size=self.query_size,
                    log_model=fsl_log_model,
                    opt=fsl_opt,
                    num_epochs=self.config.fsl_task.num_epochs,
                    save_dir=save_dir,
                    config_obj=self.config,
                )
                current_epoch_fsl_accs.append(task_acc)
                start_test_idx += self.N_way * self.query_size
                tasks_done_count += 1
                pbar.update(1)
            pbar.close()
        elif len(self.dataset.test_graphs) > 0 and total_tasks == 0:
            print(
                f"Warning: Not enough data for a full FSL evaluation task batch in epoch {epoch} (total_tasks = {total_tasks})."
            )

        mean_fsl_acc_epoch = np.mean(current_epoch_fsl_accs)
        std_fsl_acc_epoch = np.std(current_epoch_fsl_accs)
        best_fsl_acc_epoch = np.max(current_epoch_fsl_accs)

        return mean_fsl_acc_epoch, std_fsl_acc_epoch, best_fsl_acc_epoch
