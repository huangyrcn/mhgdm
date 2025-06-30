"""
Score训练函数 - 简化版本
支持双曲分数网络训练，集成采样质量监控
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import trange, tqdm
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_utils import load_config, save_config
from utils.data_utils import MyDataset
from utils.loader import load_seed, load_device, load_batch, load_model
from utils.manifolds_utils import get_manifold

# 移除 proto 相关导入
from utils.sampler import Sampler
from models.GraphVAE import GraphVAE
import ml_collections


def train_score(config, vae_checkpoint_path):
    """
    训练Score模型

    Args:
        config: 配置对象
        vae_checkpoint_path: VAE检查点路径

    Returns:
        dict: 包含训练结果和检查点路径
    """
    # 确保wandb会话干净
    close_wandb()

    # 设置基本参数
    seed = load_seed(config.seed)
    device = load_device(config)
    run_name = config.run_name

    # 初始化wandb
    _init_wandb(config)

    # 加载数据集
    dataset = MyDataset(config.data, config.fsl_task)
    train_loader, test_loader = dataset.get_loaders()

    # 加载预训练编码器
    encoder, manifold = _load_encoder(vae_checkpoint_path, device)

    # 初始化Score模型
    model_x, model_adj, optimizer_x, optimizer_adj, scheduler_x, scheduler_adj = _init_score_models(
        config, device, manifold
    )

    # 创建保存目录
    save_dir = _create_save_dir(config)

    tqdm.write(f"Score训练初始化完成: {run_name}")
    tqdm.write(f"保存目录: {save_dir}")
    tqdm.write(f"设备: {device}")

    # 主训练循环
    best_test_loss = float("inf")
    best_sample_quality = 0.0
    best_checkpoint_path = None

    progress_bar = tqdm(
        range(config.score.train.num_epochs),
        desc="Score Training",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    )

    for epoch in progress_bar:
        # 训练阶段
        train_losses = _train_epoch(
            model_x, model_adj, train_loader, optimizer_x, optimizer_adj, config, device
        )
        mean_train_x = np.mean(train_losses["x"])
        mean_train_adj = np.mean(train_losses["adj"])
        mean_train_total = mean_train_x + mean_train_adj

        # 提交训练损失到wandb
        train_log = {
            "epoch": epoch,
            "train_x_loss": mean_train_x,
            "train_adj_loss": mean_train_adj,
            "train_total_loss": mean_train_total,
            "lr_x": optimizer_x.param_groups[0]["lr"],
            "lr_adj": optimizer_adj.param_groups[0]["lr"],
        }
        wandb.log(train_log)

        # 更新学习率
        if scheduler_x:
            scheduler_x.step()
        if scheduler_adj:
            scheduler_adj.step()

        # 检查是否需要进行测试
        should_test = (epoch % config.score.train.test_interval == 0) or (
            epoch == config.score.train.num_epochs - 1
        )

        if should_test:
            # 测试阶段
            test_losses = _test_epoch(model_x, model_adj, test_loader, config, device)
            mean_test_x = np.mean(test_losses["x"])
            mean_test_adj = np.mean(test_losses["adj"])
            total_test_loss = mean_test_x + mean_test_adj

            # 采样质量评估 - 暂时注释掉
            # sample_quality = _sample_evaluation(model_x, model_adj, save_dir, config, epoch)
            sample_quality = 0.0  # 临时设置默认值

            # 提交测试损失和指标到wandb
            test_log = {
                "epoch": epoch,
                "test_x_loss": mean_test_x,
                "test_adj_loss": mean_test_adj,
                "test_total_loss": total_test_loss,
                "sample_validity": sample_quality,
            }
            wandb.log(test_log)

            # 检查是否需要保存最佳模型
            is_best_loss = total_test_loss < best_test_loss
            is_best_sample = sample_quality > best_sample_quality

            if is_best_loss:
                best_test_loss = total_test_loss
                checkpoint_path = _save_checkpoint(
                    model_x,
                    model_adj,
                    optimizer_x,
                    optimizer_adj,
                    scheduler_x,
                    scheduler_adj,
                    epoch,
                    total_test_loss,
                    sample_quality,
                    save_dir,
                    "best_loss",
                    config,
                )
                tqdm.write(f"✓ New best loss: {total_test_loss:.4f}")

            if is_best_sample:
                best_sample_quality = sample_quality
                best_checkpoint_path = _save_checkpoint(
                    model_x,
                    model_adj,
                    optimizer_x,
                    optimizer_adj,
                    scheduler_x,
                    scheduler_adj,
                    epoch,
                    total_test_loss,
                    sample_quality,
                    save_dir,
                    "best_sample",
                    config,
                )
                tqdm.write(f"✓ New best sample quality: {sample_quality:.4f}")

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "TrX": f"{mean_train_x:.2f}",
                    "TrA": f"{mean_train_adj:.2f}",
                    "TsX": f"{mean_test_x:.2f}",
                    "TsA": f"{mean_test_adj:.2f}",
                }
            )

            tqdm.write(
                f"Epoch {epoch}: Train X={mean_train_x:.4f}, Adj={mean_train_adj:.4f} | "
                f"Test X={mean_test_x:.4f}, Adj={mean_test_adj:.4f} | Sample={sample_quality:.4f}"
            )
        else:
            # 只更新进度条显示训练loss
            progress_bar.set_postfix(
                {
                    "TrX": f"{mean_train_x:.2f}",
                    "TrA": f"{mean_train_adj:.2f}",
                    "TsX": "N/A",
                    "TsA": "N/A",
                }
            )

    # 保存最终模型
    final_test_losses = _test_epoch(model_x, model_adj, test_loader, config, device)
    final_mean_test_x = np.mean(final_test_losses["x"])
    final_mean_test_adj = np.mean(final_test_losses["adj"])
    final_total_test_loss = final_mean_test_x + final_mean_test_adj
    # 暂时注释掉最终采样评估
    # final_sample_quality = _sample_evaluation(
    #     model_x, model_adj, save_dir, config, config.score.train.num_epochs - 1
    # )
    final_sample_quality = 0.0  # 临时设置默认值

    final_checkpoint_path = _save_checkpoint(
        model_x,
        model_adj,
        optimizer_x,
        optimizer_adj,
        scheduler_x,
        scheduler_adj,
        config.score.train.num_epochs - 1,
        final_total_test_loss,
        final_sample_quality,
        save_dir,
        "final",
        config,
    )

    tqdm.write(
        f"Training completed. Best test loss: {best_test_loss:.4f}, Best sample quality: {best_sample_quality:.4f}"
    )

    # 如果没有最佳采样质量检查点，则使用最终检查点
    if best_checkpoint_path is None:
        best_checkpoint_path = final_checkpoint_path

    return {
        "save_dir": save_dir,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
        "best_test_loss": best_test_loss,
        "best_sample_quality": best_sample_quality,
    }


def _init_wandb(config):
    """初始化wandb"""
    mode = "disabled" if config.debug else ("online" if config.wandb.online else "offline")

    # 从配置中获取 wandb 输出目录
    wandb_output_dir = getattr(config.wandb, "output_dir", "logs")
    wandb_dir = os.path.join(wandb_output_dir, "wandb")

    wandb.init(
        project=f"{config.wandb.project}_Score",
        entity=config.wandb.entity,
        name=f"{config.run_name}_score",
        config=OmegaConf.to_container(config, resolve=True),
        mode=mode,
        dir=wandb_dir,
    )


def _load_encoder(vae_checkpoint_path, device):
    """加载预训练编码器"""
    tqdm.write(f"Loading VAE encoder from: {vae_checkpoint_path}")
    checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)

    # 重建VAE模型
    vae_config = checkpoint["model_config"]

    # 构造GraphVAE期望的配置格式
    from types import SimpleNamespace

    model_config = SimpleNamespace()

    # 兼容两种配置格式
    if "vae" in vae_config:
        # 旧格式
        encoder_config = dict(vae_config["vae"]["encoder"])
        decoder_config = dict(vae_config["vae"]["decoder"])

        if isinstance(encoder_config, dict):
            encoder_config = SimpleNamespace(**encoder_config)
        if isinstance(decoder_config, dict):
            decoder_config = SimpleNamespace(**decoder_config)

        model_config.encoder_config = encoder_config
        model_config.decoder_config = decoder_config
        model_config.pred_node_class = vae_config["vae"]["loss"]["pred_node_class"]
        model_config.pred_edge = vae_config["vae"]["loss"]["pred_edge"]
        model_config.use_kl_loss = vae_config["vae"]["loss"]["use_kl_loss"]
        model_config.use_base_proto_loss = False
        model_config.use_sep_proto_loss = False
        model_config.latent_dim = vae_config["vae"]["encoder"]["latent_feature_dim"]
    else:
        # 新格式
        encoder_config = dict(vae_config["encoder"])
        decoder_config = dict(vae_config["decoder"])

        if isinstance(encoder_config, dict):
            encoder_config = SimpleNamespace(**encoder_config)
        if isinstance(decoder_config, dict):
            decoder_config = SimpleNamespace(**decoder_config)

        model_config.encoder_config = encoder_config
        model_config.decoder_config = decoder_config
        model_config.pred_node_class = vae_config["loss"]["pred_node_class"]
        model_config.pred_edge = vae_config["loss"]["pred_edge"]
        model_config.use_kl_loss = vae_config["loss"]["use_kl_loss"]
        model_config.use_base_proto_loss = False
        model_config.use_sep_proto_loss = False

        model_config.latent_dim = vae_config["encoder"]["latent_feature_dim"]

    model_config.device = device

    vae_model = GraphVAE(model_config)
    vae_model.load_state_dict(checkpoint["model_state_dict"])
    vae_model.to(device)
    vae_model.eval()

    # 提取编码器
    encoder = vae_model.encoder
    encoder.requires_grad_(False)
    manifold = encoder.manifold

    tqdm.write(f"✓ Encoder loaded with manifold: {manifold.__class__.__name__}")
    return encoder, manifold


def _create_save_dir(config):
    """创建保存目录"""
    if hasattr(config.paths, "score_save_dir"):
        save_dir = config.paths.score_save_dir
    else:
        save_dir = os.path.join(config.paths.save_dir, f"{config.exp_name}_score", config.timestamp)

    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _init_score_models(config, device, manifold):
    """初始化Score模型"""
    # 准备X网络配置
    # 安全地转换配置为字典
    x_config = dict(OmegaConf.to_container(config.score.x, resolve=True))

    x_config.update(
        {
            "max_feat_num": config.data.max_feat_num,
            "latent_feature_dim": config.data.max_feat_num,
            "manifold": manifold,
            "input_feature_dim": config.data.max_feat_num,
            "hidden_dim": x_config.get("nhid", 16),
        }
    )

    # 准备Adj网络配置
    # 安全地转换配置为字典
    adj_config = dict(OmegaConf.to_container(config.score.adj, resolve=True))

    adj_config.update(
        {
            "max_feat_num": config.data.max_feat_num,
            "max_node_num": config.data.max_node_num,
            "latent_feature_dim": config.data.max_feat_num,
            "manifold": manifold,
            "input_feature_dim": config.data.max_feat_num,
            "hidden_dim": adj_config.get("nhid", 16),
        }
    )

    # 创建模型
    model_x = load_model(x_config, device)
    model_adj = load_model(adj_config, device)

    # 创建优化器
    optimizer_x = optim.Adam(
        model_x.parameters(),
        lr=config.score.train.lr,
        weight_decay=config.score.train.weight_decay,
    )
    optimizer_adj = optim.Adam(
        model_adj.parameters(),
        lr=config.score.train.lr,
        weight_decay=config.score.train.weight_decay,
    )

    # 创建学习率调度器
    scheduler_x = None
    scheduler_adj = None
    if config.score.train.lr_schedule:
        scheduler_x = optim.lr_scheduler.ExponentialLR(
            optimizer_x, gamma=config.score.train.lr_decay
        )
        scheduler_adj = optim.lr_scheduler.ExponentialLR(
            optimizer_adj, gamma=config.score.train.lr_decay
        )

    tqdm.write(f"✓ Score models initialized")
    tqdm.write(f"  X model parameters: {sum(p.numel() for p in model_x.parameters()):,}")
    tqdm.write(f"  Adj model parameters: {sum(p.numel() for p in model_adj.parameters()):,}")

    return model_x, model_adj, optimizer_x, optimizer_adj, scheduler_x, scheduler_adj


def _train_epoch(model_x, model_adj, train_loader, optimizer_x, optimizer_adj, config, device):
    """训练一个epoch"""
    model_x.train()
    model_adj.train()
    losses = {"x": [], "adj": []}

    for batch in train_loader:
        x, adj, labels = load_batch(batch, device)

        # 计算损失
        loss_x = _compute_score_loss_x(model_x, x, adj, labels, device)
        loss_adj = _compute_score_loss_adj(model_adj, x, adj, labels, device)

        # X网络更新
        optimizer_x.zero_grad()
        loss_x.backward()
        torch.nn.utils.clip_grad_norm_(model_x.parameters(), config.score.train.grad_norm)
        optimizer_x.step()

        # Adj网络更新
        optimizer_adj.zero_grad()
        loss_adj.backward()
        torch.nn.utils.clip_grad_norm_(model_adj.parameters(), config.score.train.grad_norm)
        optimizer_adj.step()

        # 记录损失
        losses["x"].append(loss_x.item())
        losses["adj"].append(loss_adj.item())

    return losses


def _test_epoch(model_x, model_adj, test_loader, config, device):
    """测试一个epoch"""
    model_x.eval()
    model_adj.eval()
    losses = {"x": [], "adj": []}

    with torch.no_grad():
        for batch in test_loader:
            x, adj, labels = load_batch(batch, device)

            # 计算损失
            loss_x = _compute_score_loss_x(model_x, x, adj, labels, device)
            loss_adj = _compute_score_loss_adj(model_adj, x, adj, labels, device)

            # 记录损失
            losses["x"].append(loss_x.item())
            losses["adj"].append(loss_adj.item())

    return losses


def _compute_score_loss_x(model, x, adj, labels, device):
    """计算X网络的score matching损失"""
    from utils.graph_utils import node_flags

    # 添加噪声
    noise = torch.randn_like(x) * 0.1
    x_noisy = x + noise

    # 生成flags
    flags = node_flags(adj)

    # 生成时间步
    batch_size = x.shape[0]
    t = torch.randint(0, 100, (batch_size,), device=device).float()

    # 计算分数
    score = model(x_noisy, adj, flags, t)

    # 简化的损失计算
    loss = torch.mean((score + noise / 0.01) ** 2)
    return loss


def _compute_score_loss_adj(model, x, adj, labels, device):
    """计算Adj网络的score matching损失"""
    from utils.graph_utils import node_flags

    # 添加噪声
    noise = torch.randn_like(adj) * 0.1
    adj_noisy = adj + noise

    # 生成flags
    flags = node_flags(adj)

    # 生成时间步
    batch_size = x.shape[0]
    t = torch.randint(0, 100, (batch_size,), device=device).float()

    # 计算分数
    score = model(x, adj_noisy, flags, t)

    # 简化的损失计算
    loss = torch.mean((score + noise / 0.01) ** 2)
    return loss


def _sample_evaluation(model_x, model_adj, save_dir, config, epoch):
    """采样质量评估"""
    # 保存当前模型用于采样
    _save_current_checkpoint(model_x, model_adj, save_dir, config, epoch)

    # 创建采样器配置 - 简化版本
    try:
        # 创建一个简化的配置对象用于采样
        from types import SimpleNamespace

        # 基础采样配置
        sampling_config = SimpleNamespace()

        # 复制主要配置
        sampling_config.data = config.data
        sampling_config.seed = getattr(config, "seed", 42)
        sampling_config.device = getattr(config, "device", "auto")

        # SDE配置
        sampling_config.sde = config.score.sde

        # 采样器配置
        sampling_config.sampler = SimpleNamespace()
        sampling_config.sampler.ckp_path = os.path.join(save_dir, "current.pth")
        sampling_config.sampler.k_augment = 5  # 减少采样数量以加快速度
        sampling_config.sampler.corrector = "Langevin"
        sampling_config.sampler.n_steps = 1
        sampling_config.sampler.predictor = "Euler"
        sampling_config.sampler.scale_eps_A = 1.0
        sampling_config.sampler.scale_eps_x = 1.0
        sampling_config.sampler.snr_A = 0.25
        sampling_config.sampler.snr_x = 0.25

        # 采样参数
        sampling_config.sample = SimpleNamespace()
        sampling_config.sample.eps = 0.001
        sampling_config.sample.noise_removal = True
        sampling_config.sample.probability_flow = False
        sampling_config.sample.use_ema = True

        # 如果存在fsl_task配置，也复制过来
        if hasattr(config, "fsl_task"):
            sampling_config.fsl_task = config.fsl_task

        # 尝试进行采样
        sampler = Sampler(sampling_config)
        sample_results = sampler.sample(independent=False)

        # 提取有效性指标
        validity = sample_results.get("validity", 0.0)
        if isinstance(validity, (list, tuple)):
            validity = float(validity[0]) if len(validity) > 0 else 0.0
        elif not isinstance(validity, (int, float)):
            validity = 0.0

        return float(validity)

    except Exception as e:
        tqdm.write(f"Sampling evaluation failed: {e}")
        # 不是致命错误，返回0继续训练
        return 0.0


def _save_current_checkpoint(model_x, model_adj, save_dir, config, epoch):
    """保存当前检查点用于采样"""
    # 安全地处理配置转换
    model_config = OmegaConf.to_container(config, resolve=True)
    params_x = OmegaConf.to_container(config.score.x, resolve=True)
    params_adj = OmegaConf.to_container(config.score.adj, resolve=True)

    checkpoint = {
        "epoch": epoch,
        "model_config": model_config,
        "params_x": params_x,
        "params_adj": params_adj,
        "x_state_dict": model_x.state_dict(),
        "adj_state_dict": model_adj.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, "current.pth"))


def _save_checkpoint(
    model_x,
    model_adj,
    optimizer_x,
    optimizer_adj,
    scheduler_x,
    scheduler_adj,
    epoch,
    test_loss,
    sample_quality,
    save_dir,
    checkpoint_type,
    config,
):
    """保存检查点"""
    # 安全地处理配置转换
    model_config = OmegaConf.to_container(config, resolve=True)
    params_x = OmegaConf.to_container(config.score.x, resolve=True)
    params_adj = OmegaConf.to_container(config.score.adj, resolve=True)

    checkpoint = {
        "epoch": epoch,
        "model_config": model_config,
        "params_x": params_x,
        "params_adj": params_adj,
        "x_state_dict": model_x.state_dict(),
        "adj_state_dict": model_adj.state_dict(),
        "optimizer_x_state_dict": optimizer_x.state_dict(),
        "optimizer_adj_state_dict": optimizer_adj.state_dict(),
        "test_loss": test_loss,
        "sample_quality": sample_quality,
    }

    if scheduler_x is not None:
        checkpoint["scheduler_x_state_dict"] = scheduler_x.state_dict()
    if scheduler_adj is not None:
        checkpoint["scheduler_adj_state_dict"] = scheduler_adj.state_dict()

    save_path = os.path.join(save_dir, f"{checkpoint_type}.pth")
    torch.save(checkpoint, save_path)
    return save_path


def close_wandb():
    """确保wandb会话正确关闭"""
    try:
        if wandb.run is not None:
            wandb.finish()
    except:
        pass
