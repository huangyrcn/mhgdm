import torch
import geoopt
from geoopt import PoincareBall, Lorentz


def initialization_on_manifold(
    x_euclidean: torch.Tensor, manifold: geoopt.Manifold
) -> torch.Tensor:
    """
    Initialize a Euclidean point onto a given manifold (Lorentz or Poincare).
    Args:
        x_euclidean (Tensor): Euclidean input of shape (..., dim)
        manifold (geoopt.Manifold): An instance of Lorentz or Poincare manifold
    Returns:
        Tensor: Point on the manifold
    """
    if isinstance(manifold, geoopt.Lorentz):
        # 限制范数，避免过大带来数值不稳定
        x_euclidean = torch.nn.functional.normalize(x_euclidean, dim=-1) * 0.1
        # 拼接时间维度为 0 的切向量
        zero_time = torch.zeros_like(x_euclidean[..., :1])
        x_tangent = torch.cat([zero_time, x_euclidean], dim=-1)  # (..., d+1)
        return manifold.expmap0(x_tangent)

    elif isinstance(manifold, geoopt.PoincareBall):
        # 控制模长避免靠近边界
        x_euclidean = torch.tanh(x_euclidean) * 0.9
        return manifold.expmap0(x_euclidean)

    else:
        raise NotImplementedError(f"Unsupported manifold type: {type(manifold)}")


def is_on_lorentz_manifold(x, atol=1e-5):
    """
    判断一个向量是否在洛伦兹流形上。
    点需满足 <x, x>_L ≈ 1 且 x[..., 0] > 0
    """
    time_sq = x[..., 0] ** 2
    space_sq = x[..., 1:].pow(2).sum(dim=-1)
    lorentz_inner = time_sq - space_sq  # (...,)

    # 判断每个点是否满足 <x, x> ≈ 1
    inner_valid = torch.all(
        torch.isclose(lorentz_inner, torch.tensor(1.0, device=x.device), atol=atol)
    )
    time_positive = (x[..., 0] > 0).all()

    result = inner_valid and time_positive
    print(f"is_on_lorentz_manifold: {result}")
    return result


# 创建流形
lorentz_manifold = Lorentz(k=0.01)


# Euclidean 向量
x_e = torch.load("x_data.pt")

# 映射到 Lorentz 流形
x_l = initialization_on_manifold(x_e, lorentz_manifold)
print(x_l)
is_on_lorentz_manifold(x_l)
