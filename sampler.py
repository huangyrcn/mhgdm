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
    load_data,  # Add load_data import
    load_seed,
    load_device,
    load_model_from_ckpt,
    load_ema_from_ckpt,
    load_sampling_fn,
    load_eval_settings,
)
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
import torch.nn.functional as F


# -------- Sampler for generic graph generation tasks --------
class Sampler(object):
    def __init__(self, config):
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device(config)

    def sample(self, independent=True,protos_test=None):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config)
        self.configt = self.ckpt_dict["config"]
        self.model_x = load_model_from_ckpt(
            self.config, self.ckpt_dict["params_x"], self.ckpt_dict["x_state_dict"]
        )
        self.model_adj = load_model_from_ckpt(
            self.config, self.ckpt_dict["params_adj"], self.ckpt_dict["adj_state_dict"]
        )
        self.protos = protos_test
        self.protos= self.protos.to(self.device)
        if independent:
            if self.config.wandb.no_wandb:
                mode = "disabled"
            else:
                mode = "online" if self.config.wandb.online else "offline"
            kwargs = {
                "entity": self.config.wandb.wandb_usr,
                "name": self.config.exp_name,
                "project": self.config.wandb.project,
                "config": self.config.to_dict(),
                "settings": wandb.Settings(_disable_stats=True),
                "reinit": True,
                "mode": mode,
            }
            wandb.init(**kwargs)

        if hasattr(self.model_x, "manifold"):
            manifold = self.model_x.manifold
        else:
            manifold = None
        print("manifold:", manifold)
        if manifold is not None:
            print("k=:", manifold.k)

        load_seed(self.configt.seed)
        self.train_graph_list, self.test_graph_list = load_data(self.configt, get_graph_list=True)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)

        self.log_name = f"{self.config.exp_name}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a")

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(
                self.model_x, self.ckpt_dict["ema_x"], self.configt.train.ema
            )
            self.ema_adj = load_ema_from_ckpt(
                self.model_adj, self.ckpt_dict["ema_adj"], self.configt.train.ema
            )

            self.ema_x.copy_to(self.model_x.parameters())  # ema 模型参数复制过去
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn = load_sampling_fn(
            self.configt, self.config.sampler, self.config.sample, self.device, manifold
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(len(self.test_graph_list) / self.configt.data.batch_size)
        gen_graph_list = []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            self.init_flags = init_flags(self.train_graph_list, self.configt).to(self.device)

            x, adj = self.sampling_fn(self.model_x, self.model_adj, self.init_flags,None,self.protos)

            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

            samples_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        gen_graph_list = gen_graph_list[: len(self.test_graph_list)]
        
        
        # -------- Evaluation --------
        methods, kernels = load_eval_settings(self.config.data.name)
        result_dict = eval_graph_list(
            self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels
        )
        result_dict["mean"] = (
            result_dict["degree"] + result_dict["cluster"] + result_dict["orbit"]
        ) / 3
        logger.log(
            f"MMD_full {result_dict}"
            f"\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-"
            f"X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}"
            f"\n{self.config.saved_name}",
            verbose=False,
        )
        logger.log("=" * 100)
        if independent:
            wandb.log(result_dict, commit=True)
        # -------- Save samples --------
        save_dir = save_graph_list(self.log_folder_name, self.log_name, gen_graph_list)
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        # plot_graphs_list(graphs=sample_graph_list, title=f'{self.config.ckpt}', max_num=16,
        #                  save_dir=self.log_folder_name)
        plot_graphs_list(
            graphs=sample_graph_list,
            title=f"snr={self.config.sampler.snr_x}_scale={self.config.sampler.scale_eps_x}.png",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        return {
            "degree": result_dict["degree"],
            "cluster": result_dict["cluster"],
            "orbit": result_dict["orbit"],
            "mean": result_dict["mean"],
        }
