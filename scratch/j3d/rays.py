import torch
from typing import Union


"""
Compute the cumulative sum of w, not assuming all weight vectors sum to 1. We compute
the integral of the interior bins/weight values, not the integral of the bin edges.

Args:
    w: Tensor, which will be integrated along the last axis. w is preprocessed and
        assumed to not have any NaNs or weight vectors that sum to 0.

Returns:
    cw: Tensor, the integral of w where cw[..., -1] = 1.
"""
def integrate_weights(w) -> torch.Tensor:
    cw = torch.cumsum(w, axis=-1) / torch.sum(w, axis=-1, keepdims=True)
    # ensure that the CDF ends with 1 for all rays
    assert torch.allclose(cw[..., -1], torch.tensor(1.0, device=w.device))
    return cw


"""
Computes the weighted percentile of a batch of weight vectors by calculating the
cumulative sum of each weight vector and returning the index of the first sample
along each ray that has a value >= p.

Args:
    w: Tensor. w is a batch of weight vectors, where each weight vector stores
        the weight of each sample along the last ax is [N_rays, N_samples]. Note
        that from the volumetric rendering equations, w is not guaranteed to
        sum to 1 along the last axis; w is only guaranteed to be <= 1.
    p: float, the percentile to compute the CDF at.

Returns:
    idxs: Tensor, the index of the first sample along each ray that has a CDF
        value >= p. If all samples along a ray have a CDF value < p, then the
        function returns 0 for that ray. If p == 1.0, then the function returns
        the index of the last sample along each ray.
"""
def weighted_percentile(
    w: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    w = torch.nan_to_num(w, nan=0.0).sort(dim=-1)[0] # preprocess w
    if p == 1.0:
        return torch.argmax(w, axis=-1)
    mask = torch.sum(w, axis=-1) == 0
    if torch.all(mask): # return zero if all weights are zero
        return torch.zeros(w.shape[0], dtype=torch.int32)

    cw = integrate_weights(w[~mask])
    w[~mask] = cw
    w = (w >= p).type(torch.int32)
    return torch.argmax(w, axis=-1)


"""
Computes the cumulative distribution function of a batch of weight vectors by
first computing the probability mass function of each weight vector and then
computing the cumulative sum of each weight vector. Use this function to answer
the question "what is the percentage of xyz locations along the ray that have a
sigma value less than 400?".

Args:
    distribution: Tensor. Due to torch.histogram() not supporting axis as an
        argument, w is preprocessed to be a single discrete distribution, where
        each sample is a discrete value.
    bins: int, the number of bins to use when computing the CDF.

Returns:
    cdf: Tensor, the CDF of the distribution.
    bin_edges: Tensor, the edges of the bins used to compute the CDF.
"""
def compute_cdf(
    distribution: torch.Tensor,
    bins: Union[int, torch.Tensor]
) -> torch.Tensor:
    counts, bin_edges = extract_counts(distribution, bins)
    cdf = integrate_weights(counts).type(torch.float32)
    return cdf, bin_edges


"""
Extract histogram counts from a distribution. Calculates the PDF of some distribution
by determining number of samples in each bin.
"""
def extract_counts(
    distribution: torch.Tensor,
    bins: Union[int, torch.Tensor]
) -> torch.Tensor:
    distribution = torch.nan_to_num(distribution, nan=0.0).flatten().sort()[0]
    if isinstance(bins, torch.Tensor):
        assert torch.all(distribution >= bins[0]) and torch.all(distribution <= bins[-1]), \
            f"all values in distribution must be in the range [{bins[0]}, {bins[-1]}]"
    counts, bin_edges = torch.histogram(distribution, bins=bins)
    return counts, bin_edges


"""
Given the initial alpha/opacity value for an entire ray, and the length of the
entire ray, this function computes the ideal sigma value for the entire ray. When
considering the discretization of the ray into x number of samples, you have to take
the ideal sigma value of the entire ray and add log(x) to it to get the ideal sigma
value for each sample along the ray.
"""
def compute_ideal_sigma(alpha_init, ray_len, sample_count=None):
    alpha_init, ray_len = torch.tensor(alpha_init), torch.tensor(ray_len)
    ray_ideal_sigma = torch.log(torch.log(1.0 / (1.0 - alpha_init))) - torch.log(ray_len)
    if sample_count is not None:
        ray_ideal_sigma += torch.log(torch.tensor(sample_count))
    return ray_ideal_sigma
