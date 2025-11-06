import numpy as np

def choose_tau_from_budget(scores: list[float] | np.ndarray, review_budget: float) -> float:
    """
    Choose risk threshold τ so that ~review_budget fraction of samples would be escalated
    by risk alone. Typically computed on a validation set of fused probabilities.
    Example: review_budget=0.1 -> top 10% by risk escalated.
    """
    s = np.asarray(scores, dtype=float)
    if not (0 < review_budget < 1):
        raise ValueError("review_budget must be in (0,1)")
    return float(np.quantile(s, 1.0 - review_budget))

def choose_kappa_from_uncert(uncerts: list[float] | np.ndarray, percentile: float = 0.95) -> float:
    """
    Choose uncertainty threshold κ as a high percentile of predictive std on *benign* validation.
    """
    u = np.asarray(uncerts, dtype=float)
    return float(np.quantile(u, percentile))
