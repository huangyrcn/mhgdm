import geoopt.optim
import ml_collections
import torch
import random
import numpy as np
import subprocess
import yaml
import os

# import synthetic.model
from models.ScoreNetwork_A import (
    ScoreNetworkA,
    ScoreNetworkA_poincare,
    HScoreNetworkA,
    ScoreNetworkA_poincare_proto,
    ScoreNetworkA_euc_proto,
)
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_poincare, ScoreNetworkX_poincare_proto, ScoreNetworkX_euc_proto
from utils.sde_lib import VPSDE, VESDE, subVPSDE

from utils.losses import get_sde_loss_fn
from utils.solver import get_pc_sampler
from evaluation.mmd import gaussian, gaussian_emd
from utils.ema import ExponentialMovingAverage
import models.ScoreNetwork_X as ScoreNetwork_X
import models.ScoreNetwork_A as ScoreNetwork_A



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

def load_device(device_setting="auto", device_count=1):
    """
    Args:
        device_setting: 设备设置，可以是 "auto", "cpu", "cuda:0" 等
        device_count: 使用的设备数量
    Returns:
        selected_device: 选择的设备，可能是单个device或device列表
        device_str: 设备字符串（用于更新config.device）
    """
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

    # 如果指定了具体设备，直接返回
    if device_setting != "auto":
        if device_setting == "cpu":
            return torch.device("cpu"), "cpu"
        else:
            return torch.device(device_setting), device_setting

    # 自动选择设备
    memory_used = get_gpu_memory()
    if memory_used is None or not torch.cuda.is_available():
        print("使用 CPU。")
        return torch.device("cpu"), "cpu"

    device_count = min(device_count, torch.cuda.device_count())

    # 按显存占用排序，选择空闲的 GPU
    gpu_indices = sorted(range(len(memory_used)), key=lambda i: memory_used[i])[:device_count]

    if device_count == 1:
        selected = gpu_indices[0]
        device_str = f"cuda:{selected}"
        torch.cuda.set_device(selected)
        print(f"自动选择GPU: {device_str}")
        return torch.device(device_str), device_str
    else:
        device_str = ",".join([f"cuda:{i}" for i in gpu_indices])
        for idx in gpu_indices:
            print(f"使用GPU: cuda:{idx}")
        return gpu_indices, device_str  # ✅ 注意：返回的是 [0,1] 这样纯编号list

def load_model(config):
    """
    Load model using config object
    Args:
        config: 模型配置对象，必须包含 model_class
    """
    model_class = config.model_class
    model = None # Initialize model variable

    if model_class == 'GCN':
        import models.Encoders # Local import for encoder module
        input_feat_dim = config.input_feat_dim
        hidden_dim = config.hidden_dim
        dim = config.dim
        model = models.Encoders.GCN(
            input_feat_dim=input_feat_dim,
            hidden_dim=hidden_dim,
            dim=dim,
            enc_layers=config.enc_layers,
            layer_type=getattr(config, 'layer_type', 'GCN'),
            dropout=getattr(config, 'dropout', 0.0),
            edge_dim=getattr(config, 'edge_dim', 1),
            normalization_factor=getattr(config, 'normalization_factor', 1.0),
            aggregation_method=getattr(config, 'aggregation_method', 'sum'),
            msg_transform=getattr(config, 'msg_transform', 'linear')
        )
    elif model_class == 'HGCN':
        import models.Encoders # Local import for encoder module
        input_feat_dim = config.input_feat_dim
        hidden_dim = config.hidden_dim
        dim = config.dim
        model = models.Encoders.HGCN(
            input_feat_dim=input_feat_dim,
            hidden_dim=hidden_dim,
            dim=dim,
            enc_layers=config.enc_layers,
            layer_type=getattr(config, 'layer_type', 'HGCN'),
            dropout=getattr(config, 'dropout', 0.0),
            edge_dim=getattr(config, 'edge_dim', 1),
            normalization_factor=getattr(config, 'normalization_factor', 1.0),
            aggregation_method=getattr(config, 'aggregation_method', 'sum'),
            msg_transform=getattr(config, 'msg_transform', 'linear'),
            sum_transform=getattr(config, 'sum_transform', 'linear'),
            use_norm=getattr(config, 'use_norm', False),
            manifold=getattr(config, 'manifold', 'PoincareBall'), # Pass string name
            c=getattr(config, 'c', 1.0),    # Pass c value
            learnable_c=getattr(config, 'learnable_c', False)
        )
    elif model_class == "ScoreNetworkX":
        model = ScoreNetworkX(
            max_feat_num=config.max_feat_num,
            depth=config.depth,
            nhid=config.nhid
        )
    elif model_class == "ScoreNetworkX_euc_proto":
        model = ScoreNetworkX_euc_proto(
            max_feat_num=config.max_feat_num,
            depth=config.depth,
            nhid=config.nhid,
            proto_weight=getattr(config, 'proto_weight', 0.3)
        )
    elif model_class == "ScoreNetworkX_poincare":
        model = ScoreNetworkX_poincare(
            max_feat_num=config.max_feat_num,
            depth=config.depth,
            nhid=config.nhid,
            manifold=config.manifold,
            c=config.c,
            edge_dim=config.edge_dim,
            GCN_type=config.GCN_type
        )
    elif model_class == "ScoreNetworkX_poincare_proto":
        model = ScoreNetworkX_poincare_proto(
            max_feat_num=config.max_feat_num,
            depth=config.depth,
            nhid=config.nhid,
            manifold=config.manifold,
            c=config.c,
            edge_dim=config.edge_dim,
            GCN_type=config.GCN_type,
            proto_weight=config.proto_weight
        )
    elif model_class == "ScoreNetworkA":
        model = ScoreNetworkA(
            max_feat_num=config.max_feat_num,
            max_node_num=config.max_node_num,
            nhid=config.nhid,
            num_layers=config.num_layers,
            num_linears=config.num_linears,
            c_init=config.c_init,
            c_hid=config.c_hid,
            c_final=config.c_final,
            adim=config.adim,
            num_heads=getattr(config, 'num_heads', 4),
            conv=getattr(config, 'conv', 'GCN')
        )
    elif model_class == "ScoreNetworkA_poincare":
        model = ScoreNetworkA_poincare(
            max_feat_num=config.max_feat_num,
            max_node_num=config.max_node_num,
            nhid=config.nhid,
            num_layers=config.num_layers,
            num_linears=config.num_linears,
            c_init=config.c_init,
            c_hid=config.c_hid,
            c_final=config.c_final,
            adim=config.adim,
            num_heads=getattr(config, 'num_heads', 4),
            conv=getattr(config, 'conv', 'GCN'),
            manifold=config.manifold,
            c=config.c
        )
    elif model_class == "ScoreNetworkA_poincare_proto":
        model = ScoreNetworkA_poincare_proto(
            max_feat_num=config.max_feat_num,
            max_node_num=config.max_node_num,
            nhid=config.nhid,
            num_layers=config.num_layers,
            num_linears=config.num_linears,
            c_init=config.c_init,
            c_hid=config.c_hid,
            c_final=config.c_final,
            adim=config.adim,
            num_heads=config.num_heads,
            conv=config.conv,
            manifold=config.manifold,
            c=config.c,
            proto_weight=config.proto_weight
        )
    elif model_class == "ScoreNetworkA_euc_proto":
        model = ScoreNetworkA_euc_proto(
            max_feat_num=config.max_feat_num,
            max_node_num=config.max_node_num,
            nhid=config.nhid,
            num_layers=config.num_layers,
            num_linears=config.num_linears,
            c_init=config.c_init,
            c_hid=config.c_hid,
            c_final=config.c_final,
            adim=config.adim,
            num_heads=getattr(config, 'num_heads', 4),
            conv=getattr(config, 'conv', 'GCN'),
            manifold=getattr(config, 'manifold', None),
            proto_weight=getattr(config, 'proto_weight', 0.3)
        )
    elif model_class == "HScoreNetworkA":
        model = HScoreNetworkA(
            max_feat_num=config.max_feat_num,
            max_node_num=config.max_node_num,
            nhid=config.nhid,
            num_layers=config.num_layers,
            num_linears=config.num_linears,
            c_init=config.c_init,
            c_hid=config.c_hid,
            c_final=config.c_final,
            adim=config.adim,
            num_heads=getattr(config, 'num_heads', 4),
            conv=getattr(config, 'conv', 'GCN'),
            c=getattr(config, 'c', 1.0),
            manifold=config.manifold
        )
    else:
        raise ValueError(f"Model Name <{model_class}> is Unknown in load_model")

    return model


def load_model_optimizer(device, manifold, lr, weight_decay, lr_schedule=False, lr_decay=0.999, config=None):
    """
    Args:
        device: 设备
        manifold: 流形类型
        lr: 学习率
        weight_decay: 权重衰减
        lr_schedule: 是否使用学习率调度
        lr_decay: 学习率衰减
        config: 模型配置对象
    """
    model = load_model(config)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f"cuda:{device[0]}")
    else:
        model = model.to(device)
    
    if manifold == "Euclidean":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = geoopt.optim.RiemannianAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    scheduler = None
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

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



def load_loss_fn(sde_x_config, sde_adj_config, reduce_mean=True, eps=1e-5, manifold=None, encoder=None):
    """
    Args:
        sde_x_config: SDE配置对象，包含type, beta_min, beta_max, num_scales等参数
        sde_adj_config: SDE邻接矩阵配置对象
        reduce_mean: 是否使用平均
        eps: epsilon值
        manifold: 流形对象
        encoder: 编码器
    """
    sde_x = load_sde(sde_x_config, manifold)
    sde_adj = load_sde(sde_adj_config)

    loss_fn = get_sde_loss_fn(
        sde_x=sde_x,
        sde_adj=sde_adj,
        train=True,
        reduce_mean=reduce_mean,
        continuous=True,
        likelihood_weighting=False,
        eps=eps,
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


def load_ckpt(ckp_path, device, use_ema=False, return_ckpt=False):
    """
    Args:
        ckp_path: 检查点文件路径
        device: 设备
        use_ema: 是否使用EMA
        return_ckpt: 是否返回原始检查点
    """
    ckpt = torch.load(ckp_path, map_location=device, weights_only=False)
    print(f"{ckp_path} loaded")
    ckpt_dict = {
        "config": ckpt["model_config"],
        "params_x": ckpt["params_x"],
        "x_state_dict": ckpt["x_state_dict"],
        "params_adj": ckpt["params_adj"],
        "adj_state_dict": ckpt["adj_state_dict"],
    }
    if use_ema:
        ckpt_dict["ema_x"] = ckpt["ema_x"]
        ckpt_dict["ema_adj"] = ckpt["ema_adj"]
    if return_ckpt:
        ckpt_dict["ckpt"] = ckpt
    return ckpt_dict


def load_model_from_ckpt(device, params, state_dict):
    """
    Args:
        device: 设备
        params: 模型参数
        state_dict: 模型状态字典
    """
    model = load_model(params)
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


def load_decoder(config):
    """
    从config对象加载decoder模型
    
    Args:
        config: 包含decoder配置的对象
        
    Returns:
        实例化的decoder模型
    """
    import models.Decoders as Decoders
    
    model_class = config.model_class
    DecoderClass = getattr(Decoders, model_class)
    
    # 提取基础参数
    max_feat_num = config.max_feat_num
    hidden_dim = config.hidden_dim
    
    if model_class == 'GCN':
        # GCN Decoder特定参数
        decoder = DecoderClass(
            max_feat_num=max_feat_num,
            hidden_dim=hidden_dim,
            dec_layers=config.dec_layers,
            layer_type=getattr(config, 'layer_type', 'GCN'),
            dropout=getattr(config, 'dropout', 0.0),
            edge_dim=getattr(config, 'edge_dim', 1),
            normalization_factor=getattr(config, 'normalization_factor', 1.0),
            aggregation_method=getattr(config, 'aggregation_method', 'sum'),
            msg_transform=getattr(config, 'msg_transform', 'linear')
        )
    elif model_class == 'HGCN':
        # HGCN Decoder特定参数
        decoder = DecoderClass(
            max_feat_num=max_feat_num,
            hidden_dim=hidden_dim,
            dec_layers=config.dec_layers,
            layer_type=getattr(config, 'layer_type', 'HGCN'),
            dropout=getattr(config, 'dropout', 0.0),
            edge_dim=getattr(config, 'edge_dim', 1),
            normalization_factor=getattr(config, 'normalization_factor', 1.0),
            aggregation_method=getattr(config, 'aggregation_method', 'sum'),
            msg_transform=getattr(config, 'msg_transform', 'linear'),
            sum_transform=getattr(config, 'sum_transform', 'linear'),
            use_norm=getattr(config, 'use_norm', False),
            manifold=getattr(config, 'manifold', 'PoincareBall'),
            c=getattr(config, 'c', 1.0),
            learnable_c=getattr(config, 'learnable_c', False),
            use_centroid=getattr(config, 'use_centroid', False),
            input_manifold=getattr(config, 'input_manifold', None)
        )
    elif model_class == 'CentroidDecoder':
        # CentroidDecoder特定参数 - 使用临时导入的get_manifold
        from utils.manifolds_utils import get_manifold
        manifold_name = getattr(config, 'manifold', None)
        manifold_obj = None
        if manifold_name:
            manifold_c_val = getattr(config, 'c', 1.0 if isinstance(manifold_name, str) and ("poincare" in manifold_name.lower() or "lorentz" in manifold_name.lower()) else None)
            manifold_obj = get_manifold(manifold_name, manifold_c_val)
        decoder = DecoderClass(
            max_feat_num=max_feat_num,
            hidden_dim=hidden_dim,
            dim=config.dim,
            manifold=manifold_obj,
            dropout=getattr(config, 'dropout', 0.0)
        )
    elif model_class == 'FermiDiracDecoder':
        # FermiDiracDecoder特定参数
        decoder = DecoderClass(
            manifold=getattr(config, 'manifold', None)
        )
    elif model_class == 'Classifier':
        # Classifier特定参数
        decoder = DecoderClass(
            model_dim=config.model_dim,
            classifier_dropout=getattr(config, 'classifier_dropout', 0.0),
            classifier_bias=getattr(config, 'classifier_bias', True),
            manifold=getattr(config, 'manifold', None),
            n_classes=config.n_classes
        )
    else:
        # 基础Decoder类
        decoder = DecoderClass(
            max_feat_num=max_feat_num,
            hidden_dim=hidden_dim
        )
    
    return decoder
