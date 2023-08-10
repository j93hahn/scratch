import torch


"""
Compute the cumulative sum of w, not assuming all weight vectors sum to 1. We compute
the integral of the interior bins/weight values, not the integral of the bin edges.

Args:
    w: Tensor, which will be integrated along the last axis. w is preprocessed and
        assumed to not have any NaNs or weight vectors that sum to 0.

Returns:
    cw: Tensor, the integral of w where cw[..., -1] = 1.
"""
def integrate_weights(w):
    cw = torch.cumsum(w, axis=-1) / torch.sum(w, axis=-1, keepdims=True)
    # ensure that the CDF ends with 1 for all rays
    assert torch.allclose(cw[..., -1], torch.tensor(1.0, device=w.device))
    return cw


"""
Compute the weighted percentile of a batch of weight vectors.

Args:
    w: Tensor. w is a batch of weight vectors, where each weight vector stores
        the weight of each sample along the last axis [N_rays, N_samples]. Note
        that from the volumetric rendering equations, w is not guaranteed to
        sum to 1 along the last axis; w is only guaranteed to be <= 1.
    p: float, the percentile to compute the CDF at.

Returns:
    idxs: Tensor, the index of the first sample along each ray that has a CDF
        value >= p. If all samples along a ray have a CDF value < p, then the
        function returns 0 for that ray.
"""
def weighted_percentile(
    w,
    p: float = 0.5,
):
    w = torch.nan_to_num(w, nan=0.0)
    mask = torch.sum(w, axis=-1) == 0
    if torch.all(mask): # return zero if all weights are zero
        return torch.zeros(w.shape[0], dtype=torch.int32)

    cw = integrate_weights(w[~mask])
    w[~mask] = cw
    w = (w >= p).type(torch.int32)
    return torch.argmax(w, axis=-1)


if __name__ == '__main__':
    cw = weighted_percentile(torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.2, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.1, 0.0]
    ]))
    assert torch.all(cw == torch.tensor([0, 4, 2, 0, 0, 2]))
    print('Success!')
