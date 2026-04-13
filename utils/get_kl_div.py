import numpy as np

def get_kl_div(probs, expected_probs, eps=1e-12):
    probs = np.clip(probs, eps, 1)
    expected_probs = np.clip(expected_probs, eps, 1)
    kl_div = np.sum(probs * np.log(probs / expected_probs))
    threshold = 0.05
    return (kl_div, bool(kl_div < threshold))
