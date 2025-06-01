import torch

from utils.manifolds_utils import exp_after_transp0
from models.ScoreNetwork_X import ScoreNetworkX_poincare
from utils.sde_lib import VPSDE, VESDE, subVPSDE
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise


def get_score_fn(sde, model, train=True, continuous=True):
    """获取分数函数。"""
    if not train:
        model.eval()
    model_fn = model

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):

        def score_fn(x, adj, flags, t,labels=None, protos=None):
            if continuous:
                t_labels = t * 999
                score = model_fn(x, adj, flags, t_labels,labels, protos)
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"不支持离散时间")
            score = -score / std
            return score

    elif isinstance(sde, VESDE):

        def score_fn(x, adj, flags, t,labels=None, protos=None):
            if continuous:
                t_labels= sde.T - t
                t_labels *= sde.N - 1
                score = model_fn(x, adj, flags,t_labels, labels, protos)
            else:
                raise NotImplementedError(f"不支持离散时间")

            return score

    else:
        raise NotImplementedError(f"不支持 SDE 类 {sde.__class__.__name__}。")

    return score_fn

def get_sde_loss_fn(
    sde_x,
    sde_adj,
    train=True,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
    eps=1e-5,
    manifold=None,
    encoder=None,
):
    """返回计算 SDE loss 的闭包。"""
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kw: 0.5 * torch.sum(*args, **kw)
    )

    def loss_fn(model_x, model_adj, x, adj, labels, protos=None):
        """单批次 SDE loss（节点 + 边）"""
        flags = node_flags(adj)
        x0 = encoder(x, adj, flags).mode()          # clean representation

        # ───── 1. 噪声采样 ──────────────────────
        t = torch.rand((adj.size(0), 1, 1),
                       device=adj.device) * (sde_adj.T - eps) + eps

        z_x   = gen_noise(x0, flags, sym=False)
        μ_x, σ_x = sde_x.marginal_prob(x0, t)

        if manifold is not None:
            xt = exp_after_transp0(μ_x, σ_x * z_x, manifold)
        else:
            xt = μ_x + σ_x * z_x
        xt = mask_x(xt, flags)

        z_adj = gen_noise(adj, flags, sym=True)
        μ_a,  σ_a = sde_adj.marginal_prob(adj, t)
        at   = mask_adjs(μ_a + σ_a * z_adj, flags)

        # ───── 2. ScoreNet 前向 ─────────────────
        score_x   = model_x(xt, at, flags, t, labels, protos)
        score_adj = model_adj(xt, at, flags, t, labels, protos)

        # ───── 3. 目标分数（节点） ───────────────
        if manifold is not None:
            with torch.enable_grad():
                xt_det = xt.detach().requires_grad_(True)
                u  = manifold.logmap(μ_x, xt_det)
                v  = manifold.transp0back(μ_x, u)
                d  = manifold.dist(μ_x, xt_det, keepdim=True)
                dim = v.size(-1)

                if manifold.name == "PoincareBall":
                    x_norm2  = (xt_det ** 2).sum(-1, keepdim=True)
                    conf_fac = 2.0 / (1.0 - x_norm2 + 1e-6)
                    log_vol  = (dim - 1) * torch.log(conf_fac)
                elif manifold.name == "Lorentz":
                    sqrt_cd  = d * torch.sqrt(torch.abs(manifold.k)) + 1e-6
                    log_vol  = (dim - 1) * torch.log(
                        sqrt_cd / torch.sinh(sqrt_cd)
                    )
                else:                                  # fallback
                    sqrt_cd  = d * torch.sqrt(torch.abs(manifold.k)) + 1e-6
                    log_vol  = (dim - 1) * torch.log(
                        sqrt_cd / torch.sinh(sqrt_cd)
                    )

                logp = -0.5 * (v**2 / (σ_x**2 + 1e-8)).sum(-1, keepdim=True) \
                       + log_vol

                (egrad,) = torch.autograd.grad(
                    logp.sum(), xt_det, create_graph=False
                )
                target = manifold.egrad2rgrad(xt_det, egrad)
                target = mask_x(target, flags)
        else:
            target = -z_x / σ_x                             # 欧氏 closed-form

        # ───── 4. MSE-DSM Loss（节点） ───────────
        if manifold is not None:
            losses_x = ((score_x - target)**2).sum(-1, keepdim=True) * σ_x**2
        else:
            losses_x = (score_x * σ_x + z_x)**2              # 欧氏公式

        losses_x = reduce_op(losses_x, dim=-1)

        # ───── 5. 边分支 Loss（保持原式） ───────
        losses_a = (score_adj * σ_a + z_adj) ** 2
        losses_a = reduce_op(losses_a.reshape(losses_a.size(0), -1), dim=-1)

        # ───── 6. Normalize by valid elements ──
        adj_flags = flags.unsqueeze(2) * flags.unsqueeze(1)
        loss_x = losses_x.sum() / flags.sum()
        loss_a = losses_a.sum() / adj_flags.sum()
        return loss_x, loss_a

    return loss_fn
