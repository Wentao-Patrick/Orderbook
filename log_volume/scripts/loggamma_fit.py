import numpy as np
from scipy.stats import gamma, kstest


def fit_loggamma_positive(log_values: np.ndarray):
    data = np.asarray(log_values)
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if data.size < 10:
        raise ValueError('Not enough positive log-volume samples to fit Gamma.')

    shape_k, loc, scale_theta = gamma.fit(data, floc=0)
    ks_stat, ks_pvalue = kstest(data, lambda x: gamma.cdf(x, shape_k, loc=0, scale=scale_theta))

    return {
        'k': float(shape_k),
        'theta': float(scale_theta),
        'loc': float(loc),
        'n_fit': int(data.size),
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
    }


def loggamma_pdf(x: np.ndarray, k: float, theta: float):
    return gamma.pdf(x, a=k, loc=0, scale=theta)
