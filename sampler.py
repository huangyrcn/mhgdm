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
            if not self.config.debug:
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

                current_batch_size = adj_real.shape[0]   # 直接取当前batch大小
                shape_x = (current_batch_size, self.config.data.max_node_num, self.config.data.max_feat_num)
                shape_adj = (current_batch_size, self.config.data.max_node_num, self.config.data.max_node_num)

                x_gen, adj_gen = self.sampling_fn(self.mx, self.ma,shape_x, shape_adj, labels, self.protos)

                self.logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

                # 保存生成的图
                samples_int = quantize(adj_gen)
                gen_graph_list.extend(adjs_to_graphs(samples_int, True))

                # 保存真实的图
                adjs_real_int = quantize(adj_real)
                graph_ref_list.extend(adjs_to_graphs(adjs_real_int,True))

        # 只采样，不评估
        if not need_eval:
            return gen_graph_list

        # --------- 评估 ---------
        methods, kernels = load_eval_settings()
        result_dict = eval_graph_list(
            graph_ref_list, gen_graph_list, methods=methods, kernels=kernels
        )
        result_dict["mean"] = (
            result_dict["degree"] + result_dict["cluster"] + result_dict["orbit"]
        ) / 3
        print(result_dict)
        self.logger.log(
            f"MMD_full {result_dict}"
            f"\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-"
            f"X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}"
            f"\n{self.config.run_name}",
            verbose=False,
        )

        self.logger.log("=" * 100)
        if self.independent:
            wandb.log(result_dict, commit=True)
        return result_dict
