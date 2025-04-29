import os
import time
import pickle
import math

import geoopt
import ml_collections
import numpy as np
import torch
import wandb
from models.HVAE import HVAE
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import (
    load_ckpt,
    load_data,  # 添加 load_data 导入
    load_seed,
    load_device,
    load_model_from_ckpt,
    load_ema_from_ckpt,
    load_sampling_fn,
    load_eval_settings,
    load_batch,
)
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
import torch.nn.functional as F
from utils.protos_utils import compute_protos_from
from utils.graph_utils import node_flags


# -------- 通用图生成任务的采样器 --------
class mySampler(object):
    """
    使用预训练模型生成图的采样器类。
    """

    def __init__(self, config):
        """
        使用配置和设备初始化采样器。

        Args:
            config: 包含采样参数的配置对象。
        """
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device(config)  # 加载指定的设备（CPU 或 GPU）

    def sample(self, independent=True):
        """
        根据加载的模型和配置生成图样本。

        Args:
            independent (bool): 如果为 True，则独立初始化 wandb 用于日志记录。
            protos_test: 测试原型（如果提供）。

        Returns:
            dict: 包含评估指标（度、聚类、轨道、均值）的字典。
        """
        # -------- 加载检查点 --------
        # 加载包含模型状态和配置的检查点字典
        self.ckpt_dict = load_ckpt(self.config)
        # 从检查点加载训练配置
        self.configt = self.ckpt_dict["config"]
        # 从检查点加载特征模型 (model_x)
        self.model_x = load_model_from_ckpt(
            self.config, self.ckpt_dict["params_x"], self.ckpt_dict["x_state_dict"]
        )
        # 从检查点加载邻接模型 (model_adj)
        self.model_adj = load_model_from_ckpt(
            self.config, self.ckpt_dict["params_adj"], self.ckpt_dict["adj_state_dict"]
        )
        # 分配测试原型并将其移动到设备

        # -------- 初始化 WandB (可选) --------
        if independent:
            # 根据配置确定 wandb 模式
            if self.config.wandb.no_wandb:
                mode = "disabled"
            else:
                mode = "online" if self.config.wandb.online else "offline"
            # 定义 wandb 初始化参数
            kwargs = {
                "entity": self.config.wandb.wandb_usr,
                "name": self.config.exp_name,
                "project": self.config.wandb.project,
                "config": self.config.to_dict(),
                "settings": wandb.Settings(_disable_stats=True),
                "reinit": True,
                "mode": mode,
            }
            # 初始化 wandb 运行
            wandb.init(**kwargs)

        # -------- 流形设置 --------
        # 检查特征模型是否具有流形属性
        if hasattr(self.model_x, "manifold"):
            manifold = self.model_x.manifold
        else:
            manifold = None
        # 打印正在使用的流形 # Removed debug print
        if manifold is not None:
            pass  # Removed debug print

        # -------- 加载数据和设置日志记录 --------
        # 从训练配置加载随机种子
        load_seed(self.configt.seed)
        # 加载训练和测试图列表
        self.train_graph_list, self.test_graph_list = load_data(self.configt, get_graph_list=True)

        # 设置采样日志目录和文件夹名称
        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)

        # 定义日志文件名
        self.log_name = f"{self.config.exp_name}-sample"
        # 初始化日志记录器实例
        logger = Logger(str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a")

        # 检查日志文件是否存在，如果不存在，则写入初始日志信息
        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            start_log(logger, self.configt)  # 记录启动配置
            train_log(logger, self.configt)  # 记录训练配置详情
        # 记录采样特定配置
        sample_log(logger, self.config)

        # -------- 加载模型 --------
        # 如果已配置，则加载指数移动平均 (EMA) 模型
        if self.config.sample.use_ema:
            # 加载特征模型的 EMA 状态
            self.ema_x = load_ema_from_ckpt(
                self.model_x, self.ckpt_dict["ema_x"], self.configt.train.ema
            )
            # 加载邻接模型的 EMA 状态
            self.ema_adj = load_ema_from_ckpt(
                self.model_adj, self.ckpt_dict["ema_adj"], self.configt.train.ema
            )

            # 将 EMA 参数复制到主模型
            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        # 根据配置加载适当的采样函数
        self.sampling_fn = load_sampling_fn(
            self.configt, self.config.sampler, self.config.sample, self.device, manifold
        )

        # -------- 生成样本 --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")  # 记录生成种子
        # 加载生成种子
        load_seed(self.config.sample.seed)  # 加载生成种子

        # 根据批量大小计算所需的采样轮数
        num_sampling_rounds = math.ceil(len(self.test_graph_list) / self.configt.data.batch_size)
        gen_graph_list = []  # 初始化列表以存储生成的图
        # 循环进行采样轮次
        for r in range(num_sampling_rounds):
            t_start = time.time()  # 记录本轮开始时间

            # 根据训练数据初始化标志（例如，节点计数）
            self.init_flags = init_flags(self.train_graph_list, self.configt).to(self.device)

            # 使用加载的函数和模型执行采样
            x, adj = self.sampling_fn(
                self.model_x, self.model_adj, self.init_flags, None, self.protos
            )

            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")  # 记录本轮耗时
            # 量化生成的邻接矩阵
            samples_int = quantize(adj)
            # 将量化的邻接矩阵转换为图对象
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        # 修剪生成的列表以匹配测试集的大小
        gen_graph_list = gen_graph_list[: len(self.test_graph_list)]

        # -------- 评估 --------
        # 根据数据集名称加载评估方法和核函数
        methods, kernels = load_eval_settings(self.config.data.name)
        # 使用指定的指标评估生成的图与测试集
        result_dict = eval_graph_list(
            self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels
        )
        # 计算平均 MMD 分数
        result_dict["mean"] = (
            result_dict["degree"] + result_dict["cluster"] + result_dict["orbit"]
        ) / 3
        # 记录评估结果和采样器配置详情
        logger.log(
            f"MMD_full {result_dict}"
            f"\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-"
            f"X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}"
            f"\n{self.config.saved_name}",
            verbose=False,
        )
        logger.log("=" * 100)  # 日志中的分隔线
        # 如果独立运行，则将结果记录到 wandb
        if independent:
            wandb.log(result_dict, commit=True)

        # -------- 保存样本 --------
        # 将生成的图列表保存到文件
        save_dir = save_graph_list(self.log_folder_name, self.log_name, gen_graph_list)
        # 加载保存的图（可选，用于验证或绘图）
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        # 绘制生成图的子集并保存绘图
        # plot_graphs_list(graphs=sample_graph_list, title=f'{self.config.ckpt}', max_num=16,
        #                  save_dir=self.log_folder_name)
        plot_graphs_list(
            graphs=sample_graph_list,
            title=f"snr={self.config.sampler.snr_x}_scale={self.config.sampler.scale_eps_x}.png",  # 标题包含采样器参数
            max_num=16,  # 要绘制的最大图数
            save_dir=self.log_folder_name,  # 保存绘图的目录
        )
        # 返回评估指标
        return {
            "degree": result_dict["degree"],
            "cluster": result_dict["cluster"],
            "orbit": result_dict["orbit"],
            "mean": result_dict["mean"],
        }


class Sampler(object):
    """
    使用预训练模型生成图的采样器类。
    """

    def __init__(self, mode, config):
        assert mode in {"ckpt", "mem"}, "mode 只能是 'ckpt' 或 'mem'"

        self.config = config
        self.device = load_device(config)

        # ---------- score_ckpt 分支 ----------
        if mode == "ckpt":
            self.independent = True
            score_ckpt = torch.load(
                config.score_ckpt_path, map_location=self.device, weights_only=False
            )
            self.configt = ml_collections.ConfigDict(score_ckpt["model_config"])
            self.mx = load_model_from_ckpt(
                config, score_ckpt["params_x"], score_ckpt["x_state_dict"]
            )
            self.ma = load_model_from_ckpt(
                config, score_ckpt["params_adj"], score_ckpt["adj_state_dict"]
            )

            if config.sample.use_ema:
                load_ema_from_ckpt(self.mx, score_ckpt["ema_x"], self.configt.train.ema).copy_to(
                    self.mx.parameters()
                )
                load_ema_from_ckpt(self.ma, score_ckpt["ema_adj"], self.configt.train.ema).copy_to(
                    self.ma.parameters()
                )
            for p in self.mx.parameters():
                p.requires_grad = False
            for p in self.ma.parameters():
                p.requires_grad = False
            self.mx.eval()
            self.ma.eval()
            # 从训练配置加载随机种子
            load_seed(self.configt.seed)
            _, self.dataloader = load_data(self.configt, get_graph_list=False)

            ae_ckpt = torch.load(
                self.config.ae_ckpt_path, map_location=self.config.device, weights_only=False
            )
            AE_state_dict = ae_ckpt["ae_state_dict"]
            AE_config = ml_collections.ConfigDict(ae_ckpt["model_config"])
            ae = HVAE(AE_config)
            ae.load_state_dict(AE_state_dict, strict=False)
            for p in ae.encoder.parameters():
                p.requires_grad = False
            self.encoder = ae.encoder.to(self.device).eval()
            del ae.decoder

            self.protos = compute_protos_from(self.encoder, self.dataloader, self.device)

            if type(self.mx.manifold) != type(self.encoder.manifold):
                raise ValueError("模型流形不匹配")
            else:
                self.manifold = self.encoder.manifold
                # 设置采样日志目录和文件夹名称
            self.log_folder_name, self.log_dir, _ = set_log(self.config, is_train=False)
            self.sampling_fn = load_sampling_fn(
                self.configt, config.sampler, config.sample, self.device, self.manifold
            )
            # 定义日志文件名
            self.log_name = f"{self.config.exp_name}-sample"
            # 初始化日志记录器实例
            self.logger = Logger(str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a")

            # 检查日志文件是否存在，如果不存在，则写入初始日志信息
            if not check_log(self.log_folder_name, self.log_name):
                self.logger.log(f"{self.log_name}")
                start_log(self.logger, self.configt)  # 记录启动配置
                train_log(self.logger, self.configt)  # 记录训练配置详情
            # 记录采样特定配置
            sample_log(self.logger, self.config)
        # ---------- mem 分支 ----------
        else:  # mode == "mem"
            self.independent = False
            self.mx = config.model_x.to(self.device).eval()
            self.ma = config.model_adj.to(self.device).eval()
            self.encoder = config.encoder.to(self.device).eval()
            self.dataloader = config.dataloader
            self.protos = compute_protos_from(self.encoder, self.dataloader, self.device)
            self.manifold = self.mx.manifold
            load_seed(self.config.sample.seed)
            self.sampling_fn = load_sampling_fn(
                config, config.sampler, config.sample, self.device, self.manifold
            )
            self.logger = config.logger
            sample_log(self.logger, self.config)

    def sample(self, need_eval=True):
        """
        按batch采样，每个batch独立生成。
        Args:
            need_eval (bool): 是否评估生成图。
        Returns:
            dict or list: 评估指标 or 生成的图列表。
        """
        if self.independent:
            mode = "disabled"
            if not self.config.wandb.no_wandb:
                mode = "online" if self.config.wandb.online else "offline"
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.run_name,
                config=self.config.to_dict(),
                settings=wandb.Settings(_disable_stats=True),
                mode=mode,
            )
        self.logger.log(f"GEN SEED: {self.config.sample.seed}")

        gen_graph_list = []
        graph_ref_list = []
        with torch.no_grad():  # 采样阶段禁用梯度计算
            for r, batch in enumerate(self.dataloader):
                x_real, adj_real, labels = load_batch(batch, self.device)
                t_start = time.time()

                self.flags = node_flags(adj_real).to(self.device)

                x_gen, adj_gen = self.sampling_fn(self.mx, self.ma, self.flags,labels, self.protos)

                self.logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

                # 保存生成的图
                samples_int = quantize(adj_gen)
                gen_graph_list.extend(adjs_to_graphs(samples_int, is_discrete=True))

                # 保存真实的图
                adjs_real_int = quantize(adj_real)
                graph_ref_list.extend(adjs_to_graphs(adjs_real_int, is_discrete=True))

        # 只采样，不评估
        if not need_eval:
            return gen_graph_list

        # --------- 评估 ---------
        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict = eval_graph_list(
            graph_ref_list, gen_graph_list, methods=methods, kernels=kernels
        )
        result_dict["mean"] = (
            result_dict["degree"] + result_dict["cluster"] + result_dict["orbit"]
        ) / 3
        self.logger.log(
            f"MMD_full {result_dict}"
            f"\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-"
            f"X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}"
            f"\n{self.config.saved_name}",
            verbose=False,
        )
        self.logger.log("=" * 100)
        if self.independent:
            wandb.log(result_dict, commit=True)
        return result_dict
