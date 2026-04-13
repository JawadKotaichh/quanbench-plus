def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator used in code-generation evals.
    n: number of samples
    c: number of correct samples
    k: cutoff
    """
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if c >= k:
        return 1.0
    if k > n:
        k = n

    # 1 - C(n-c, k)/C(n, k)
    # Compute ratio as product to avoid overflow:
    # C(n-c, k)/C(n, k) = Π_{i=0..k-1} (n-c-i)/(n-i)
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod
