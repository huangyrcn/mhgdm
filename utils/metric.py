import torch

def EuclideanDistances(a, b):
    """
    Calculates pairwise Euclidean distances between two sets of tensors.

    Args:
        a (torch.Tensor): Tensor of shape (m, d)
        b (torch.Tensor): Tensor of shape (n, d)

    Returns:
        torch.Tensor: Tensor of shape (m, n) containing pairwise distances.
    """
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m -> [m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n -> [1, n]
    bt = b.t()
    # Clamp minimum value to avoid sqrt of negative numbers due to floating point errors
    dist_sq = sum_sq_a + sum_sq_b - 2 * a.mm(bt)
    return torch.sqrt(torch.clamp(dist_sq, min=0.0))


