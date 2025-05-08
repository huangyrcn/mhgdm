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

    def __init__(self, config):


        self.config = config
        self.device = load_device(config)

        # ---------- score_ckpt 分支 ----------
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
        
        # 如果config中有dataloader就使用config中的，否则加载新的
        if hasattr(self.config, 'dataloader') and self.config.dataloader is not None:
            self.dataloader = self.config.dataloader
        else:
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


    def sample(self, need_eval=True):
        """
        按batch采样，每个batch独立生成。
        Args:
            need_eval (bool): 是否评估生成图。
        Returns:
            dict or DataLoader: 评估指标 or 增强后的数据加载器。
        """
        # if self.independent:
        #     mode = "disabled"
        #     if not self.config.debug:
        #         mode = "online" if self.config.wandb.online else "offline"
        #     wandb.init(
        #         project=self.config.wandb.project,
        #         entity=self.config.wandb.entity,
        #         name=self.config.run_name,
        #         config=self.config.to_dict(),
        #         settings=wandb.Settings(_disable_stats=True),
        #         mode=mode,
        #     )
        # self.logger.log(f"GEN SEED: {self.config.sample.seed}")

        gen_graph_list = []
        graph_ref_list = []
        
        # 存储k-augmented数据
        k_augment = self.config.sample.k_augment
        augmented_x_list = []
        augmented_adj_list = []
        augmented_labels_list = []
        
        with torch.no_grad():  # 采样阶段禁用梯度计算
            for r, batch in enumerate(self.dataloader):
                x_real, adj_real, labels = load_batch(batch, self.device)
                t_start = time.time()

                current_batch_size = adj_real.shape[0]   # 直接取当前batch大小
                shape_x = (current_batch_size, self.config.data.max_node_num, self.config.data.max_feat_num)
                shape_adj = (current_batch_size, self.config.data.max_node_num, self.config.data.max_node_num)
                
                # 为每个原始样本生成k个增强样本
                for _ in range(k_augment):
                    x_gen, adj_gen = self.sampling_fn(self.mx, self.ma, shape_x, shape_adj, labels, self.protos)
                    
                    # 添加到增强数据集
                    augmented_x_list.append(x_gen)
                    augmented_adj_list.append(adj_gen)
                    augmented_labels_list.append(labels)
                
                # 添加原始数据到增强数据集
                augmented_x_list.append(x_real)
                augmented_adj_list.append(adj_real)
                augmented_labels_list.append(labels)

                self.logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

                # 保存生成的图和原始图（用于评估）
                if need_eval:
                    samples_int = quantize(adj_gen)
                    gen_graph_list.extend(adjs_to_graphs(samples_int, True))
                    
                    adjs_real_int = quantize(adj_real)
                    graph_ref_list.extend(adjs_to_graphs(adjs_real_int, True))

        # 创建增强数据集的数据加载器
        augmented_x = torch.cat(augmented_x_list, dim=0)
        augmented_adj = torch.cat(augmented_adj_list, dim=0)
        augmented_labels = torch.cat(augmented_labels_list, dim=0)
        
        # 创建数据集和数据加载器
        from torch.utils.data import TensorDataset, DataLoader
        augmented_dataset = TensorDataset(augmented_x, augmented_adj, augmented_labels)
        batch_size = self.dataloader.batch_size if hasattr(self.dataloader, 'batch_size') else 32
        augmented_dataloader = DataLoader(
            augmented_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.dataloader.num_workers if hasattr(self.dataloader, 'num_workers') else 0
        )
        
        # 返回增强的数据加载器（如果不需要评估）
        if not need_eval:
            self.logger.log(f"Created k-augmented dataloader with k={k_augment}, total samples: {len(augmented_dataset)}")
            return augmented_dataloader

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
            
        # 同时返回评估结果和增强的数据加载器
        result_dict["augmented_dataloader"] = augmented_dataloader
        return result_dict
