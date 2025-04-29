import torch

from utils.manifolds_utils import exp_after_transp0
from models.ScoreNetwork_X import ScoreNetworkX_poincare
from utils.sde_lib import VPSDE, VESDE, subVPSDE
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise


def get_score_fn(sde, model, train=True, continuous=True):
    """获取分数函数。

    Args:
        sde: SDE（随机微分方程）对象。
        model: 分数模型。
        train: 是否处于训练模式。
        continuous: 是否使用连续时间。

    Returns:
        分数函数。
    """
    if not train:
        model.eval() # 如果不是训练模式，则将模型设置为评估模式
    model_fn = model

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):

        def score_fn(x, adj, flags, t,labels=None, protos=None): # 保留 protos 参数
            # 通过标准差缩放神经网络输出并翻转符号
            if continuous:
                t_labels = t * 999 # 缩放时间标签
                # 将 protos 传递给 model_fn 调用
                score = model_fn(x, adj, flags, t_labels,labels, protos)
                # 获取边际概率的标准差
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"不支持离散时间")
            # if not isinstance(model_fn, ScoreNetworkX_poincare):    # 待办事项
            # 分数取反并除以标准差
            score = -score / std
            return score

    elif isinstance(sde, VESDE):

        def score_fn(x, adj, flags, t,labels=None, protos=None): # 保留 protos 参数
            if continuous:
                # 计算时间标签
                t_labels= sde.T - t
                t_labels *= sde.N - 1
                # 将 protos 传递给 model_fn 调用
                score = model_fn(x, adj, flags,t_labels, labels, protos)
            else:
                raise NotImplementedError(f"不支持离散时间")

            return score

    else:
        raise NotImplementedError(f"不支持 SDE 类 {sde.__class__.__name__}。")

    return score_fn


def get_sde_loss_fn(
    sde_x, # 特征的 SDE
    sde_adj, # 邻接矩阵的 SDE
    train=True, # 是否处于训练模式
    reduce_mean=False, # 是否对损失进行平均
    continuous=True, # 是否使用连续时间
    likelihood_weighting=False, # 是否使用似然加权
    eps=1e-5, # 用于避免数值问题的小常数
    manifold=None, # 流形对象（如果适用）
    encoder=None, # 编码器（如果适用）
):
    """获取 SDE 损失函数。

    Args:
        sde_x: 特征的 SDE 对象。
        sde_adj: 邻接矩阵的 SDE 对象。
        train: 是否处于训练模式。
        reduce_mean: 是否对损失进行平均。
        continuous: 是否使用连续时间。
        likelihood_weighting: 是否使用似然加权。
        eps: 用于避免数值问题的小常数。
        manifold: 流形对象（如果适用）。
        encoder: 编码器（如果适用）。

    Returns:
        SDE 损失函数。
    """
    # 输出的分数与数据结果相同，需要 reduce_op 来转为一个数
    # 定义损失聚合操作（求和或求平均）
    reduce_op = (
        torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model_x, model_adj, x, adj, labels, protos=None):
        """计算 SDE 损失。

        Args:
            model_x: 特征分数模型。
            model_adj: 邻接矩阵分数模型。
            x: 特征张量。
            adj: 邻接矩阵张量。
            labels: 标签（如果适用）。
        Returns:
            特征损失和邻接矩阵损失的元组。
        """
        # 获取节点标志
        flags = node_flags(adj)
        # 如果提供了编码器，则使用编码器获取特征
        if encoder is not None:
            posterior = encoder(x, adj, flags)
            x = posterior.mode()
        else:
            # 如果提供了流形，则将特征映射到切空间
            if manifold is not None:
                x = manifold.expmap0(x)

        # 获取特征和邻接矩阵的分数函数
        score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=train, continuous=continuous)

        # 随机采样时间 t
        t = torch.rand((adj.shape[0], 1, 1), device=adj.device) * (sde_adj.T - eps) + eps
       
        # region 生成扰动数据
        
        # 特征扰动数据
        z_x = gen_noise(x, flags, sym=False) # 生成特征噪声
        mean_x, std_x = sde_x.marginal_prob(x, t) # 计算特征的边际均值和标准差

        # 如果使用流形，则在切空间中添加噪声并映射回流形
        if manifold is not None:
            perturbed_x = exp_after_transp0(mean_x, std_x * z_x, manifold)
        else:
            # 否则，直接添加高斯噪声
            perturbed_x = mean_x + std_x * z_x
        perturbed_x = mask_x(perturbed_x, flags) # 应用节点掩码
       
        # 结构扰动数据
        z_adj = gen_noise(adj, flags, sym=True) # 生成邻接矩阵噪声（对称）
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t) # 计算邻接矩阵的边际均值和标准差
        perturbed_adj = mean_adj + std_adj * z_adj # 添加噪声
        perturbed_adj = mask_adjs(perturbed_adj, flags) # 应用邻接矩阵掩码

        # endregion 

        # region 计算score
        # 计算扰动数据的分数
        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t,labels,protos)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t,labels,protos)
        # endregion

        # 计算损失
        if not likelihood_weighting:
            # 如果不使用似然加权
            if manifold is not None:
                # 在流形上计算损失
                with torch.enable_grad():
                    xt = perturbed_x.detach()
                    xt.requires_grad = True
                    # 计算对数映射和并行传输
                    u = manifold.logmap(mean_x, xt)
                    v = manifold.transp0back(mean_x, u)
                    dim = v.size(-1)
                    dist = manifold.dist(mean_x, xt, keepdim=True)
                    sqrt_c_dist = (
                        dist * torch.sqrt(torch.abs(manifold.k)) + 1e-6
                    )  # 添加 eps 以避免 nan

                    # 计算对数概率密度
                    logp = -1 * v**2 / (2 * std_x**2).sum(-1, keepdims=True) + (
                        dim - 1
                    ) * torch.log(sqrt_c_dist / torch.sinh(sqrt_c_dist))
                    
                    # 计算目标分数（对数概率密度的梯度）
                    (target,) = torch.autograd.grad(logp.sum(), xt)
                    target = mask_x(target, flags) # 应用节点掩码
                # 计算分数和目标之间的平方误差
                losses_x = torch.square(score_x - target)
            else:
                # 在欧氏空间计算损失
                losses_x = torch.square(score_x * std_x + z_x)  # 计算分数匹配损失
            losses_x = reduce_op(losses_x, dim=-1) # 聚合特征损失
            # 计算邻接矩阵的分数匹配损失
            losses_adj = torch.square(score_adj * std_adj + z_adj)
        else:
            # 似然加权（likelihood weighting）损失计算
            # g2_x 和 g2_adj 分别为特征和结构的扩散系数平方
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            # 计算score和噪声的加权平方误差
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            # 按batch聚合损失并乘以扩散系数平方，实现似然加权
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj

        # 计算邻接矩阵标志
        adj_flags = flags.unsqueeze(2) * flags.unsqueeze(1)  # b*n*n

        # 返回归一化的损失
        return (
            torch.sum(losses_x.view(-1)) / flags.sum(), # 特征损失除以节点数
            torch.sum(losses_adj.view(-1)) / adj_flags.sum(), # 邻接矩阵损失除以边数（或可能的边数）
        )


    return loss_fn
