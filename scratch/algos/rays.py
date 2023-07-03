import numpy as np


"""
compute the weights histogram of multiple rays --

vectorized wrapper function that computes the weights histogram of every ray in the batch simultaneously
"""
def histogram_percentile_val(
    weights: np.ndarray,
    sigmas: np.ndarray,
    xyz_samples: np.ndarray,
    percentile: float = 0.5,
):
    # weights: N_rays x N_samples
    # sigmas: N_rays x N_samples
    # xyz_samples: N_rays x N_samples x 3
    _idxs = np.zeros(weights.shape[0], dtype=np.int32)
    _weights = np.zeros(weights.shape[0], dtype=np.float32)
    _sigmas = np.zeros(weights.shape[0], dtype=np.float32)
    _xyz_locs = np.zeros((weights.shape[0], 3), dtype=np.float32)

    # mask will store the indices of the rays that have not yet been processed
    # as false; once a ray has been processed, its index will be set to true
    mask = np.zeros(weights.shape[0], dtype=bool)
    _weights_sum = weights.sum(axis=-1)

    # if the sum of the weights is 0, the ray passed through empty space; apply a
    # mask to that ray as it will not contribute to the final image
    if np.any(_weights_sum == 0):
        mask[_weights_sum == 0] = True
        if mask.sum() == weights.shape[0]:
            return _idxs, _weights, _sigmas, _xyz_locs

    # return the maximum weight if it is >= 50% of the total sum of the weights; this
    # value must divide the weights histogram into two equal parts
    np.seterr(divide='ignore', invalid='ignore')    # ignore divide by zero warnings
    if np.any((weights[~mask].max(axis=-1) / _weights_sum[~mask]) >= 0.5):
        _wmax = np.nan_to_num(weights.max(axis=-1) / _weights_sum, nan=0.0)
        _wmax = (_wmax >= 0.5) & (~mask)
        _idxs[_wmax] = weights.argmax(axis=-1)[_wmax]
        _weights[_wmax] = weights[_wmax, _idxs[_wmax]]
        _sigmas[_wmax] = sigmas[_wmax, _idxs[_wmax]]
        _xyz_locs[_wmax] = xyz_samples[_wmax, _idxs[_wmax]]

        # apply a mask to the rays that have been processed
        mask[_wmax] = True
        if mask.sum() == weights.shape[0]:
            return _idxs, _weights, _sigmas, _xyz_locs

    # normalize the weights of each ray to sum 1 and compute its cumulative distribution function
    weights_cum = np.nan_to_num(weights / _weights_sum[..., None], nan=0.0)
    weights_cum = np.cumsum(weights_cum, axis=-1)

    # extract the first weight value at the specified percentile of the CDF for each ray
    weights_cum = (weights_cum >= percentile)
    _wpercentile = np.argmax(weights_cum, axis=-1) # argmax returns the first index where the condition is met

    # store the index, weight, and xyz location of the given ray
    _idxs[~mask] = _wpercentile[~mask]
    _weights[~mask] = weights[~mask, _idxs[~mask]]
    _sigmas[~mask] = sigmas[~mask, _idxs[~mask]]
    _xyz_locs[~mask] = xyz_samples[~mask, _idxs[~mask]]

    # turn divide by zero warnings back on
    np.seterr(divide='warn', invalid='warn')

    # return the indices, weights, and xyz locations of the rays
    return _idxs, _weights, _sigmas, _xyz_locs
