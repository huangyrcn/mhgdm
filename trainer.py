import os
import time

import ml_collections
import torch.nn.functional as F  # 确保导入了 F

import wandb
from tqdm import trange
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
from utils.logger import (
    Logger,
    set_log,
    start_log,
    train_log,
)
from utils.manifolds_utils import (
    get_manifold,
)
from layers.Decoders import Classifier,LogReg  # 使用 LogReg 替代 Classifier


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        (
            self.log_folder_name,
            self.log_dir,
            self.ckpt_dir,
        ) = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device(self.config)
        self.run_name = self.config.run_name

    def train_ae(self):
        ts = self.config.timestamp

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
            dir=os.path.join("logs", "wandb"),  # 👈 将 wandb 日志写入 logs/wandb 目录
        )

        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # -------- Load data models, optimizers --------
        (
            self.train_loader,
            self.test_loader,
        ) = load_data(self.config)
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

        logger = Logger(
            str(
                os.path.join(
                    self.log_dir,
                    f"{self.run_name}.log",
                )
            ),
            mode="a",
        )
        logger.log(
            f"{self.run_name}",
            verbose=False,
        )
        start_log(logger, self.config)
        train_log(logger, self.config)

        # -------- Training --------

        best_mean_test_loss = 1e10

        for epoch in trange(
            0,
            (self.config.train.num_epochs),
            desc="[Epoch]",
            leave=False,
        ):

            self.total_train_loss = []
            self.total_test_loss = []
            self.test_kl_loss = []
            self.test_edge_loss = []
            self.test_rec_loss = []
            self.test_proto_loss = []  # 新增：记录测试时的原型损失
            t_start = time.time()

            # train

            self.model.train()

            for (
                step,
                batch,
            ) in enumerate(self.train_loader):
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
            for (
                _,
                test_batch,
            ) in enumerate(self.test_loader):
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
            mean_test_proto_loss = np.mean(self.test_proto_loss)  # 计算平均原型损失
            if (
                "HGCN"
                in [
                    self.config.model.encoder,
                    self.config.model.decoder,
                ]
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
                "current_loss": mean_total_test_loss,
            }

            # 按固定间隔保存epoch检查点
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
                os.makedirs(
                    os.path.dirname(f"{save_dir}/epoch_{epoch}.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/epoch_{epoch}.pth",
                )

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

            # 保存最终模型
            if epoch == self.config.train.num_epochs - 1:
                os.makedirs(
                    os.path.dirname(f"{save_dir}/final.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/final.pth",
                )

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
            self.manifold = get_manifold(
                self.config.model.manifold,
                self.config.model.c,
            )
        else:
            checkpoint = torch.load(
                self.config.model.ae_path,
                map_location=self.config.device,
                weights_only=False,
            )
            AE_state_dict = checkpoint["ae_state_dict"]
            AE_config = ml_collections.ConfigDict(checkpoint["model_config"])
            AE_config.model.dropout = 0
            ae = HVAE(AE_config)
            ae.load_state_dict(
                AE_state_dict,
                strict=False,
            )
            for (
                name,
                param,
            ) in ae.named_parameters():
                if "encoder" in name or "decoder" in name:
                    param.requires_grad = False
            Encoder = ae.encoder.to(self.device)
            self.manifold = Encoder.manifold

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
            dir=os.path.join("logs", "wandb"),  # 👈 将 wandb 日志写入 logs/wandb 目录
        )

        print("\033[91m" + f"{self.run_name}" + "\033[0m")

        # -------- Load data models, optimizers, ema --------
        (
            self.train_loader,
            self.test_loader,
        ) = load_data(self.config)
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

        logger = Logger(
            str(
                os.path.join(
                    self.log_dir,
                    f"{self.run_name}.log",
                )
            ),
            mode="a",
        )
        logger.log(
            f"{self.run_name}",
            verbose=False,
        )
        start_log(logger, self.config)
        train_log(logger, self.config)

        # region compute protos
        # 计算元训练集的 protos

        protos_train = compute_protos_from(
            Encoder,
            self.train_loader,
            self.device,
        )
        protos_test = compute_protos_from(
            Encoder,
            self.test_loader,
            self.device,
        )

        # end region

        self.loss_fn = load_loss_fn(
            self.config,
            self.manifold,
            encoder=Encoder,
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
                "current_loss": total_test_loss,
            }

            # 按固定间隔保存epoch检查点
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
                os.makedirs(
                    os.path.dirname(f"{save_dir}/epoch_{epoch}.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/epoch_{epoch}.pth",
                )

            # 判断并保存最佳模型
            if total_test_loss < best_mean_test_loss:
                best_mean_test_loss = total_test_loss
                save_dict["best_loss"] = best_mean_test_loss
                os.makedirs(
                    os.path.dirname(f"{save_dir}/best.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/best.pth",
                )

            # 保存最终模型
            if epoch == self.config.train.num_epochs - 1:
                os.makedirs(
                    os.path.dirname(f"{save_dir}/final.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/final.pth",
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

        # Load dataset
        from utils.data_utils import MyDataset

        self.dataset = MyDataset(self.config)

        # Load pretrained encoder
        if self.config.model.ae_path is None:
            raise ValueError("No autoencoder path specified in config")

        checkpoint = torch.load(
            self.config.model.ae_path, map_location=self.device, weights_only=False
        )
        AE_state_dict = checkpoint["ae_state_dict"]
        AE_config = ml_collections.ConfigDict(checkpoint["model_config"])
        AE_config.model.dropout = 0
        ae = HVAE(AE_config)
        ae.load_state_dict(AE_state_dict, strict=False)

        # Freeze encoder parameters
        for name, param in ae.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

        self.model = ae.encoder.to(self.device)
        self.model.eval()

        # 初始化必要的参数
        self.N_way = self.config.fsl_task.N_way
        self.K_shot = self.config.fsl_task.K_shot
        self.query_size = self.config.fsl_task.query_size

        # 初始化分类器和优化器
        ft_in = self.config.model.hidden_dim
        nb_classes = self.N_way
        self.log = LogReg(ft_in, nb_classes).to(self.device)
        self.opt = torch.optim.Adam(self.log.parameters(), lr=self.config.train.lr)
        self.xent = torch.nn.CrossEntropyLoss()

        # 初始化logger
        logger = Logger(
            str(
                os.path.join(
                    self.log_dir,
                    f"{self.run_name}.log",
                )
            ),
            mode="a",
        )
        logger.log(
            f"{self.run_name}",
            verbose=False,
        )
        start_log(logger, self.config)
        train_log(logger, self.config)

        # 创建保存目录
        save_dir = (
            f"./checkpoints/{self.config.data.name}/{self.config.exp_name}/{self.config.timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs("./savepoint", exist_ok=True)

        # 初始化最佳准确率
        best_mean_acc = 0.0
        best_std = 0.0
        best_test_accs = []

        # Sample test tasks
        test_accs = []
        start_test_idx = 0
        t_start = time.time()
        
        # 计算总任务数量，用于进度条显示
        total_tasks = (len(self.dataset.test_graphs) -self.K_shot * self.dataset.test_classes_num)/(self.N_way * self.query_size)
        
        # 使用tqdm显示整体训练进度
        from tqdm import tqdm
        task_progress = tqdm(total=total_tasks, desc="Training models", position=0)

        while (
            start_test_idx
            < len(self.dataset.test_graphs) - self.K_shot * self.dataset.test_classes_num
        ):
            test_acc = self.train_one_step(epoch=0, test_idx=start_test_idx)
            test_accs.append(test_acc)
            start_test_idx += self.N_way * self.query_size
            
            # 更新整体进度条
            task_progress.update(1)
            task_progress.set_postfix({"last_acc": f"{test_acc:.4f}", "mean_acc": f"{sum(test_accs)/len(test_accs):.4f}"})

            # 计算当前准确率
            mean_acc = sum(test_accs) / len(test_accs)
            std = np.array(test_accs).std()

            # 构建保存信息
            save_dict = {
                "model_config": self.config.to_dict(),
                "log_state_dict": self.log.state_dict(),
                "mean_acc": mean_acc,
                "std": std,
                "test_accs": test_accs,
                "best_mean_acc": best_mean_acc,
                "current_mean_acc": mean_acc,
            }


            # 判断并保存最佳模型
            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_std = std
                best_test_accs = test_accs.copy()
                save_dict["best_mean_acc"] = best_mean_acc
                os.makedirs(
                    os.path.dirname(f"{save_dir}/best.pth"),
                    exist_ok=True,
                )
                torch.save(
                    save_dict,
                    f"{save_dir}/best.pth",
                )

        
        # 关闭进度条
        task_progress.close()


        print("Mean Test Acc {:.4f}  Std {:.4f}".format(mean_acc, std))
        print("Best Mean Test Acc {:.4f}  Std {:.4f}".format(best_mean_acc, best_std))

        return mean_acc, std

    def train_one_step(self, epoch, test_idx):
        """
        Train or evaluate on a single task.

        :param epoch: Current epoch number
        :param test_idx: Index of the test task
        :return: Accuracy for the task
        """
        self.model.eval()

        # Sample one task using dataset's method
        first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
        current_task = self.dataset.sample_one_task(
            self.dataset.test_tasks,
            first_N_class_sample,
            K_shot=self.K_shot,
            query_size=self.query_size,
            test_start_idx=test_idx,
        )

        # Get support set data
        support_x = current_task["support_set"]["x"].to(self.device)
        support_adj = current_task["support_set"]["adj"].to(self.device)
        support_label = current_task["support_set"]["label"].to(self.device)

        # 创建支持集的数据加载器
        support_dataset = torch.utils.data.TensorDataset(support_x, support_adj, support_label)
        support_loader = torch.utils.data.DataLoader(
            support_dataset, batch_size=len(support_x), shuffle=False  # 可以调整批次大小
        )
        
        # 使用 sampler 扩充 support_loader
        # 创建一个临时配置对象，用于初始化 Sampler
        from sampler import Sampler
        import ml_collections
        import copy
        
        # 创建采样器配置
        sampler_config = copy.deepcopy(self.config)
        
        # 设置采样器要使用的 dataloader
        sampler_config.dataloader = support_loader
        
        sampler = Sampler(sampler_config)
        augmented_support_loader = sampler.sample(need_eval=False)
        
        # 将原始支持集与增强数据结合，而不是替换
        # 从两个DataLoader中提取数据
        augmented_x = augmented_support_loader.dataset.tensors[0]
        augmented_adj = augmented_support_loader.dataset.tensors[1]
        augmented_label = augmented_support_loader.dataset.tensors[2]
        
        # 确保所有张量都在同一个设备上 (self.device)
        augmented_x = augmented_x.to(self.device)
        augmented_adj = augmented_adj.to(self.device)
        augmented_label = augmented_label.to(self.device)
        
        # 打印标签信息，用于调试
        print(f"augmented_label: {augmented_label.cpu().numpy()}")
        print(f"Augmented labels range: min={augmented_label.min().item()}, max={augmented_label.max().item()}")
        print(f"Unique labels in augmented data: {torch.unique(augmented_label).cpu().numpy()}")
        print(f"Expected N_way: {self.N_way}")
        
        # 确保标签在有效范围内 (0 到 N_way-1)
        if augmented_label.max() >= self.N_way:
            print(f"WARNING: Found labels outside expected range. Clamping values from range [{augmented_label.min()}, {augmented_label.max()}] to [0, {self.N_way-1}].")
            augmented_label = torch.clamp(augmented_label, 0, self.N_way - 1)
            print(f"After clamping - unique labels: {torch.unique(augmented_label).cpu().numpy()}")
        
        # 创建组合数据集
        combined_dataset = torch.utils.data.TensorDataset(
            augmented_x, augmented_adj, augmented_label
        )
        
        # 创建新的DataLoader
        combined_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=len(support_x),  # 使用原始支持集的大小作为批次大小
            shuffle=True  # 打乱数据顺序
        )
        
        # 使用组合后的数据加载器
        support_loader = combined_loader
        print(f"Successfully combined support set: original size={len(support_dataset)}, augmented size={len(augmented_support_loader.dataset)}, combined size={len(combined_dataset)}")


        # 初始化分类器训练参数
        self.log.train()  # 设置为训练模式
        # best_loss = 1e9
        # wait = 0
        # patience = 10

        # 对每个批次依次处理：先计算embedding，然后针对该批次进行多轮训练
        for batch_x, batch_adj, batch_label in support_loader:
            # 为整个批次计算 node_mask
            node_masks = torch.stack([node_flags(adj) for adj in batch_adj])

            with torch.no_grad():
                # 计算当前批次的嵌入
                posterior = self.model(batch_x, batch_adj, node_masks)
                graph_embs = posterior.mode()

                # 处理维度，确保得到正确的图嵌入
                if graph_embs.dim() == 3 and graph_embs.size(1) == 1:
                    graph_embs = graph_embs.squeeze(1)

                # 对每个图的节点嵌入取平均，得到图级嵌入
                batch_embeddings = graph_embs.mean(dim=1)  # 在节点维度上平均

            # 对当前批次的嵌入训练多轮
            from tqdm import trange
            
            # 使用配置文件中的训练轮次
            num_epochs = self.config.train.num_epochs
            
            # 使用tqdm显示训练进度
            for _ in trange(num_epochs, desc="Training batch", leave=False):
                self.opt.zero_grad()
                logits = self.log(batch_embeddings)
                
                # 计算损失
                loss = F.cross_entropy(logits, batch_label.long())
                
                # 添加L2正则化
                l2_reg = torch.tensor(0.0).to(self.device)
                for param in self.log.parameters():
                    l2_reg += torch.norm(param)
                loss = loss + 0.1 * l2_reg
                
                # 反向传播和优化
                loss.backward()
                self.opt.step()
                
                # 保存最新的模型权重
                torch.save(self.log.state_dict(), f"./savepoint/{self.config.data.name}_lr.pkl")
        
        # 评估阶段
        self.log.eval()

        # Get query set data
        query_x = current_task["query_set"]["x"].to(self.device)
        query_adj = current_task["query_set"]["adj"].to(self.device)
        query_label = current_task["query_set"]["label"].to(self.device)

        # 创建查询集的数据加载器
        query_dataset = torch.utils.data.TensorDataset(query_x, query_adj, query_label)

        # 处理可能的append_count
        query_len = query_label.shape[0]
        effective_len = query_len
        if current_task["append_count"] != 0:
            effective_len = query_len - current_task["append_count"]
            query_dataset = torch.utils.data.TensorDataset(
                query_x[:effective_len], query_adj[:effective_len], query_label[:effective_len]
            )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=effective_len, shuffle=False  # 可以调整批次大小
        )

        # Process query set as batches
        query_data = []
        for batch_x, batch_adj, batch_label in query_loader:
            # 为整个批次计算 node_mask
            node_masks = torch.stack([node_flags(adj) for adj in batch_adj])

            with torch.no_grad():
                # 直接处理整个批次
                posterior = self.model(batch_x, batch_adj, node_masks)
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

        # Calculate accuracy
        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_labels).float() / query_labels.shape[0]

        test_acc = acc.cpu().numpy()

        return test_acc
