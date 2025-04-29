import torch
import numpy as np
import abc
from tqdm import trange

from losses import get_score_fn
from utils.manifolds_utils import (
    exp_after_transp0,
    transp0back_after_logmap,
    mobius_scalar_mul,
    mobius_sub,
    mobius_add,
)
from utils.graph_utils import mask_adjs, mask_x, gen_noise, node_flags
from utils.sde_lib import VPSDE, subVPSDE

import sys
class Predictor(abc.ABC):
    """预测器算法的抽象类。"""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # 计算反向 SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        """更新函数，需要在子类中实现。"""
        pass


class Corrector(abc.ABC):
    """校正器算法的抽象类。"""

    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr # 信噪比
        self.scale_eps = scale_eps # 噪声缩放因子
        self.n_steps = n_steps # 校正步数

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        """更新函数，需要在子类中实现。"""
        pass


class EulerMaruyamaPredictor(Predictor):
    """欧拉-丸山预测器。"""
    def __init__(self, obj, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj # 'x' 或 'adj'

    def update_fn(self, x, adj, flags, t):
        """执行一步欧拉-丸山预测。"""
        dt = -1.0 / self.rsde.N # 时间步长

        if self.obj == "x":
            z = gen_noise(x, flags, sym=False) # 生成噪声
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False) # 计算漂移和扩散项
            if self.sde.hyp: # 如果使用双曲空间
                m = self.sde.manifold
                x_mean = m.expmap(x, drift * dt) # 计算均值 (测地线)
                x = exp_after_transp0(x_mean, diffusion * np.sqrt(-dt) * z, m) # 添加噪声 (平行传输后)
            else: # 欧氏空间
                x_mean = x + drift * dt # 计算均值
                x = x_mean + diffusion * np.sqrt(-dt) * z # 添加噪声
            return x, x_mean

        elif self.obj == "adj":
            z = gen_noise(adj, flags) # 生成噪声
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True) # 计算漂移和扩散项
            adj_mean = adj + drift * dt # 计算均值
            adj = adj_mean + diffusion * np.sqrt(-dt) * z # 添加噪声

            return adj, adj_mean

        else:
            raise NotImplementedError(f"对象 {self.obj} 尚不支持。")


class ReverseDiffusionPredictor(Predictor):
    """反向扩散预测器。"""
    def __init__(self, obj, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj # 'x' 或 'adj'

    def update_fn(self, x, adj, flags, t):
        """执行一步反向扩散预测。"""

        if self.obj == "x":
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False) # 获取离散化的漂移和扩散项
            z = gen_noise(x, flags, sym=False) # 生成噪声
            if self.sde.hyp: # 如果使用双曲空间
                m = self.sde.manifold
                x_mean = m.expmap(x, -f) # 计算均值 (测地线)
                x = exp_after_transp0(x_mean, G * z, m) # 添加噪声 (平行传输后)
            else: # 欧氏空间
                x_mean = x - f # 计算均值
                x = x_mean + G * z # 添加噪声
            return x, x_mean

        elif self.obj == "adj":
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True) # 获取离散化的漂移和扩散项
            z = gen_noise(adj, flags) # 生成噪声
            adj_mean = adj - f # 计算均值
            adj = adj_mean + G * z # 添加噪声
            return adj, adj_mean

        else:
            raise NotImplementedError(f"对象 {self.obj} 尚不支持。")


class NoneCorrector(Corrector):
    """一个空的校正器，什么也不做。"""

    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)
        self.obj = obj
        pass

    def update_fn(self, x, adj, flags, t, labels=None, protos=None):
        """不执行任何校正。"""
        if self.obj == "x":
            return x, x
        elif self.obj == "adj":
            return adj, adj
        else:
            raise NotImplementedError(f"对象 {self.obj} 尚不支持。")


class LangevinCorrector(Corrector):
    """朗之万动力学校正器。"""
    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)
        self.obj = obj # 'x' 或 'adj'

    def update_fn(self, x, adj, flags, t,labels=None, protos=None):
        """执行朗之万动力学校正。"""
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps
        m = sde.manifold

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()  # (200,1,1)
            flat_timestep = timestep.view(-1)            # (200)
            alpha = sde.alphas.to(t.device).index_select(0, flat_timestep)  # (200)
            alpha = alpha.view(t.shape)  # reshape回 (200,1,1)
        else:
            alpha = torch.ones_like(t)

        if self.obj == "x":
            x_mean = None
            for i in range(n_steps): # 执行 n_steps 校正
                grad = score_fn(x, adj, flags, t,labels,protos) # 计算得分函数 (梯度)
                noise = gen_noise(x, flags, sym=False) # 生成噪声
                noise_norm = torch.norm(noise, dim=-1).mean() # 计算噪声范数
                if sde.hyp: # 如果使用双曲空间
                    grad_norm = m.norm(x, grad, dim=-1).mean()  # 计算梯度范数
                    step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha # 计算步长
                    x_mean = m.expmap(x, step_size * grad) # 更新均值 (测地线)
                    x = exp_after_transp0(x_mean, torch.sqrt(step_size * 2) * noise * seps, m) # 添加噪声 (平行传输后)
                else: # 欧氏空间
                    grad_norm = torch.norm(grad, dim=-1).mean() # 计算梯度范数
                    step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha # 计算步长
                    x_mean = x + step_size * grad # 更新均值
                    x = x_mean + torch.sqrt(step_size * 2) * noise * seps # 添加噪声
            return x, x_mean

        elif self.obj == "adj":
            adj_mean = None
            for i in range(n_steps): # 执行 n_steps 校正
                grad = score_fn(x, adj, flags, t,labels,protos) # 计算得分函数 (梯度)
                noise = gen_noise(adj, flags) # 生成噪声
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean() # 计算梯度范数
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean() # 计算噪声范数
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha # 计算步长
                adj_mean = adj + step_size * grad # 更新均值
                adj = adj_mean + torch.sqrt(step_size * 2) * noise * seps # 添加噪声
            return adj, adj_mean

        else:
            raise NotImplementedError(f"对象 {self.obj} 尚不支持")


# -------- PC sampler --------
def get_pc_sampler(
    sde_x, # x 的 SDE
    sde_adj, # adj 的 SDE
    device, # 设备 (CPU/GPU)
    predictor="Euler", # 预测器类型 ("Euler" 或 "Reverse")
    corrector="None", # 校正器类型 ("Langevin" 或 "None")
    probability_flow=False, # 是否使用概率流 ODE
    continuous=False, # 是否使用连续时间得分匹配
    denoise=True, # 是否在最后一步去噪
    eps=1e-3, # 最小时间步
    config_module=None, # 配置模块
):

    def pc_sampler(model_x, model_adj, shape_x,shape_adj,labels, protos=None):
        # 获取配置参数
        n_steps = config_module.n_steps # 校正器步数
        snr_x = config_module.snr_x # x 的信噪比
        scale_eps_x = config_module.scale_eps_x # x 的噪声缩放因子

        snr_A = config_module.snr_A # adj 的信噪比
        scale_eps_A = config_module.scale_eps_A # adj 的噪声缩放因子

        # 获取 x 和 adj 的得分函数
        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

        # 选择预测器和校正器类型
        predictor_fn = (
            ReverseDiffusionPredictor if predictor == "Reverse" else EulerMaruyamaPredictor
        )
        corrector_fn = LangevinCorrector if corrector == "Langevin" else NoneCorrector

        # 初始化 x 的预测器和校正器
        predictor_obj_x = predictor_fn("x", sde_x, score_fn_x, probability_flow)
        corrector_obj_x = corrector_fn("x", sde_x, score_fn_x, snr_x, scale_eps_x, n_steps)

        # 初始化 adj 的预测器和校正器
        predictor_obj_adj = predictor_fn("adj", sde_adj, score_fn_adj, probability_flow)
        corrector_obj_adj = corrector_fn("adj", sde_adj, score_fn_adj, snr_A, scale_eps_A, n_steps)

        with torch.no_grad(): # 禁用梯度计算
            # -------- 初始化样本 --------
            x = sde_x.prior_sampling(shape_x).to(device) # 从先验分布采样 x
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device) # 从先验分布采样 adj (对称)
            
            flags = node_flags(adj) # 获取标志位
            x = mask_x(x, flags) # 根据标志位掩码 x
            adj = mask_adjs(adj, flags) # 根据标志位掩码 adj
            diff_steps = sde_adj.N # 获取扩散步数
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device) # 生成时间步

            # -------- 反向扩散过程 --------
            for i in trange(0, (diff_steps), desc="[Sampling]", position=1, leave=False): # 迭代时间步
                t = timesteps[i] # 当前时间步
                vec_t = torch.ones((shape_adj[0], 1, 1), device=t.device) * t # 创建时间向量

                _x = x # 保存当前 x

                # 校正步骤
                x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t,labels,protos) # 校正 x
                adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t,labels,protos) # 校正 adj

                _x = x # 保存校正后的 x

                # 预测步骤
                x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t) # 预测 x
                adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t) # 预测 adj
                # print(i,':',torch.argmax(x[0],dim=-1)) # 打印调试信息 (可选)
            print(" ") # 打印空行
            # 返回最终结果，根据 denoise 参数决定是否返回去噪后的结果
            return (x_mean if denoise else x), (adj_mean if denoise else adj)

    return pc_sampler # 返回采样器函数


# -------- S4 solver --------
def S4_solver(
    sde_x, # x 的 SDE
    sde_adj, # adj 的 SDE
    shape_x, # x 的形状
    shape_adj, # adj 的形状
    predictor="None", # 预测器类型 (当前未使用)
    corrector="None", # 校正器类型 (当前未使用)
    snr=0.1, # 校正步骤的信噪比
    scale_eps=1.0, # 校正步骤的噪声缩放因子
    n_steps=1, # 校正步数 (当前未使用)
    probability_flow=False, # 是否使用概率流 ODE (当前未使用)
    continuous=False, # 是否使用连续时间得分匹配
    denoise=True, # 是否在最后一步去噪
    eps=1e-3, # 最小时间步
    device="cuda", # 设备 (CPU/GPU)
):
    """获取 S4 (Score-based Stochastic Solver) 求解器。"""

    def s4_solver(model_x, model_adj, init_flags):
        """S4 求解器的内部函数。"""

        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous) # 获取 x 的得分函数
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous) # 获取 adj 的得分函数

        with torch.no_grad():
            # -------- Initial sample --------
            x = sde_x.prior_sampling(shape_x).to(device) # 从先验分布采样 x
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device) # 从先验分布采样 adj (对称)
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
            dt = -1.0 / diff_steps

            # -------- Rverse diffusion process --------
            for i in trange(0, (diff_steps), desc="[Sampling]", position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt / 2)

                # -------- Score computation --------
                score_x = score_fn_x(x, adj, flags, vec_t)
                score_adj = score_fn_adj(x, adj, flags, vec_t)

                Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
                Sdrift_adj = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

                # -------- Correction step --------
                timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                if isinstance(sde_x, VPSDE):
                    alpha = sde_x.alphas.to(vec_t.device)[timestep]
                else:
                    alpha = torch.ones_like(vec_t)

                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * score_x
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                if isinstance(sde_adj, VPSDE):
                    alpha = sde_adj.alphas.to(vec_t.device)[timestep]  # VP
                else:
                    alpha = torch.ones_like(vec_t)  # VE
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * score_adj
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

                # -------- Prediction step --------
                x_mean = x
                adj_mean = adj
                mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)
                x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                x = x + Sdrift_x * dt
                adj = adj + Sdrift_adj * dt

                mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
                x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                x_mean = mu_x
                adj_mean = mu_adj
            print(" ")
            return (x_mean if denoise else x), (adj_mean if denoise else adj), 0

    return s4_solver # 返回求解器函数
