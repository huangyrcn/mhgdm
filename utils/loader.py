import geoopt.optim
import ml_collections
import torch
import random
import numpy as np
import subprocess
import yaml
import os

# import synthetic.model
from models.HVAE import HVAE
from models.ScoreNetwork_A import (
    ScoreNetworkA_poincare,
    HScoreNetworkA,
    ScoreNetworkA_poincare_proto,
)
from models.ScoreNetwork_X import ScoreNetworkX_poincare, ScoreNetworkX_poincare_proto
from utils.sde_lib import VPSDE, VESDE, subVPSDE

from losses import get_sde_loss_fn
from solver import get_pc_sampler
from evaluation.mmd import gaussian, gaussian_emd
from utils.ema import ExponentialMovingAverage

from utils.data_utils import Dataset
from .data_utils import load_data

import subprocess


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def load_device(config):
    def get_gpu_memory():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            return [int(x) for x in output.strip().split("\n")]
        except Exception as e:
            print(f"无法获取 GPU 内存信息: {e}")
            return None

    memory_used = get_gpu_memory()
    if memory_used is None or not torch.cuda.is_available():
        print("使用 CPU。")
        config.device = "cpu"
        return torch.device("cpu")

    device_count = getattr(config, "device_count", 1)
    device_count = min(device_count, torch.cuda.device_count())

    # 按显存占用排序，选择空闲的 GPU
    gpu_indices = sorted(range(len(memory_used)), key=lambda i: memory_used[i])[:device_count]

    if device_count == 1:
        selected = gpu_indices[0]
        config.device = f"cuda:{selected}"
        torch.cuda.set_device(selected)
        print(f"自动选择GPU: cuda:{selected}")
        return torch.device(config.device)
    else:
        config.device = ",".join([f"cuda:{i}" for i in gpu_indices])
        for idx in gpu_indices:
            print(f"使用GPU: cuda:{idx}")
        return gpu_indices  # ✅ 注意：返回的是 [0,1] 这样纯编号list

def load_model(config, params):
    """
    Load model using current configuration structure
    """
    model_type = params["model_type"]
    # 根据模型类型选择配置
    if model_type == "ScoreNetworkX_poincare":
        model = ScoreNetworkX_poincare(**params)
    elif model_type == "x_poincare_proto":
        model = ScoreNetworkX_poincare_proto(**params)
    elif model_type == "ScoreNetworkA_poincare":
        model = ScoreNetworkA_poincare(**params)
    elif model_type == "adj_poincare_proto":
        model = ScoreNetworkA_poincare_proto(**params)
    elif model_type == "HScoreNetworkA":
        model = HScoreNetworkA(**params)
    elif model_type == "ae":
        model = HVAE(config)  # HVAE 需要完整的 config
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")

    return model


def load_model_optimizer(config, params):
    config_train = config.train
    device = config.device
    model = load_model(config, params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f"cuda:{device[0]}")
    else:
        model = model.to(device)
    if config.model.manifold == "Euclidean":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config_train.lr,
            weight_decay=config_train.weight_decay,
        )
    else:
        optimizer = geoopt.optim.RiemannianAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config_train.lr,
            weight_decay=config_train.weight_decay,
        )
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)

    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_batch(batch, device):
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device

    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    labels_b = batch[2].to(device_id)
    return x_b, adj_b, labels_b


def load_sde(config_sde, manifold=None):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == "VP":
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales, manifold=manifold)
    elif sde_type == "VE":
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales, manifold=manifold)
    elif sde_type == "subVP":
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales, manifold=manifold)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde



def load_loss_fn(config,manifold=None, encoder=None):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x, manifold)
    sde_adj = load_sde(config.sde.adj)

    loss_fn = get_sde_loss_fn(
        sde_x=sde_x,
        sde_adj=sde_adj,
        train=True,
        reduce_mean=reduce_mean,
        continuous=True,
        likelihood_weighting=False,
        eps=config.train.eps,
        manifold=manifold,
        encoder=encoder,
    )
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, device, manifold):
    """
    构建采样函数。
    Args:
        config_train: 训练相关配置，包含SDE参数。
        config_module: 采样模块配置，包含predictor/corrector等。
        config_sample: 采样参数配置，包含采样细节如probability_flow、noise_removal等。
        device: 设备（如'cpu'或'cuda'）。
        manifold: 流形对象。
    Returns:
        sampling_fn: 采样函数。
    """
    # 加载特征和邻接矩阵的SDE
    sde_x = load_sde(config_train.sde.x, manifold)
    sde_adj = load_sde(config_train.sde.adj)

    # 构建采样函数
    sampling_fn =  get_pc_sampler(
        sde_x=sde_x,
        sde_adj=sde_adj,
        device=device,
        predictor=config_module.predictor,
        corrector=config_module.corrector,
        probability_flow=config_sample.probability_flow,
        continuous=True,
        denoise=config_sample.noise_removal,
        eps=config_sample.eps,
        config_module=config_module,
    )
    return sampling_fn


# def load_model_params(config, manifold=None):
#     config_m = config.model
#     max_feat_num = config.data.max_feat_num

#     if "GMH" in config_m.x:
#         params_x = {
#             "model_type": config_m.x,
#             "max_feat_num": max_feat_num,
#             "depth": config_m.depth,
#             "nhid": config_m.nhid,
#             "num_linears": config_m.num_linears,
#             "c_init": config_m.c_init,
#             "c_hid": config_m.c_hid,
#             "c_final": config_m.c_final,
#             "adim": config_m.adim,
#             "num_heads": config_m.num_heads,
#             "conv": config_m.conv,
#         }
#     elif "poincare" in config_m.x:
#         params_x = {
#             "model_type": config_m.x,
#             "max_feat_num": max_feat_num,
#             "depth": config_m.depth,
#             "nhid": config_m.nhid,
#             "manifold": manifold,
#             "edge_dim": config_m.edge_dim,
#             "GCN_type": config_m.GCN_type,
#         }
#     elif "poincare_proto" in config_m.x:
#         params_x = {
#             "model_type": config_m.x,
#             "max_feat_num": max_feat_num,
#             "depth": config_m.depth,
#             "nhid": config_m.nhid,
#             "manifold": manifold,
#             "edge_dim": config_m.edge_dim,
#             "GCN_type": config_m.GCN_type,
#             "proto_weight": config_m.proto_weight,
#         }
#     else:
#         params_x = {
#             "model_type": config_m.x,
#             "max_feat_num": max_feat_num,
#             "depth": config_m.depth,
#             "nhid": config_m.nhid,
#         }

#     if "poincare" in config_m.adj:
#         params_adj = {
#             "model_type": config_m.adj,
#             "max_feat_num": max_feat_num,
#             "max_node_num": config.data.max_node_num,
#             "nhid": config_m.nhid,
#             "num_layers": config_m.num_layers,
#             "num_linears": config_m.num_linears,
#             "c_init": config_m.c_init,
#             "c_hid": config_m.c_hid,
#             "c_final": config_m.c_final,
#             "adim": config_m.adim,
#             "num_heads": config_m.num_heads,
#             "conv": config_m.conv,
#             "manifold": manifold,
#         }
#     elif "poincare_proto" in config_m.adj:
#         params_adj = {
#             "model_type": config_m.adj,
#             "max_feat_num": max_feat_num,
#             "max_node_num": config.data.max_node_num,
#             "nhid": config_m.nhid,
#             "num_layers": config_m.num_layers,
#             "num_linears": config_m.num_linears,
#             "c_init": config_m.c_init,
#             "c_hid": config_m.c_hid,
#             "c_final": config_m.c_final,
#             "adim": config_m.adim,
#             "num_heads": config_m.num_heads,
#             "conv": config_m.conv,
#             "manifold": manifold,
#             "proto_weight": config_m.proto_weight,
#         }
#     elif "HScoreNetworkA" == config_m.adj:
#         params_adj = {
#             "model_type": config_m.adj,
#             "max_feat_num": max_feat_num,
#             "max_node_num": config.data.max_node_num,
#             "nhid": config_m.nhid,
#             "num_layers": config_m.num_layers,
#             "num_linears": config_m.num_linears,
#             "c_init": config_m.c_init,
#             "c_hid": config_m.c_hid,
#             "c_final": config_m.c_final,
#             "adim": config_m.adim,
#             "num_heads": config_m.num_heads,
#             "conv": config_m.conv,
#             "manifold": manifold,
#         }
#     else:
#         params_adj = {
#             "model_type": config_m.adj,
#             "max_feat_num": max_feat_num,
#             "max_node_num": config.data.max_node_num,
#             "nhid": config_m.nhid,
#             "num_layers": config_m.num_layers,
#             "num_linears": config_m.num_linears,
#             "c_init": config_m.c_init,
#             "c_hid": config_m.c_hid,
#             "c_final": config_m.c_final,
#             "adim": config_m.adim,
#             "num_heads": config_m.num_heads,
#             "conv": config_m.conv,
#         }
#     return params_x, params_adj


def load_ckpt(config, return_ckpt=False):

    path = config.sampler.ckp_path
    ckpt = torch.load(path, map_location=config.device,weights_only=False)
    print(f"{path} loaded")
    ckpt_dict = {
        "config": ckpt["model_config"],
        "params_x": ckpt["params_x"],
        "x_state_dict": ckpt["x_state_dict"],
        "params_adj": ckpt["params_adj"],
        "adj_state_dict": ckpt["adj_state_dict"],
    }
    if config.sample.use_ema:
        ckpt_dict["ema_x"] = ckpt["ema_x"]
        ckpt_dict["ema_adj"] = ckpt["ema_adj"]
    if return_ckpt:
        ckpt_dict["ckpt"] = ckpt
    return ckpt_dict


def load_model_from_ckpt(config, params, state_dict):
    model = load_model(config, params)
    device = config.device
    if "module." in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f"cuda:{device[0]}")
    else:
        model = model.to(device)
    return model


def load_eval_settings():
    # Settings for generic graph generation
    methods = ["degree", "cluster", "orbit", "spectral"]
    kernels = {
        "degree": gaussian_emd,
        "cluster": gaussian_emd,
        "orbit": gaussian,
        "spectral": gaussian_emd,
    }
    return methods, kernels
