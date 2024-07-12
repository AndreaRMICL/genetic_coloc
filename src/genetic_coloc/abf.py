import numpy as np


def compute_log_abf(beta: np.ndarray, se: np.ndarray, se_prior: float) -> np.ndarray:
    """Calculates the log Approximate Bayes Factor (ABF).

    Args:
        beta: The beta coefficients from the regression analysis.
        se: The standard errors of the beta coefficients.
        se_prior: The prior standard error. The recommended values
            are 0.15*SD of trait for quantitative traits and 0.20
            for binary traits.

    Returns:
        The log ABF values.

    Example:
        >>> import numpy as np
        >>> from genetic_coloc import compute_log_abf
        >>> beta = np.array([0.1, 0.2, 0.3])
        >>> se = np.array([0.01, 0.02, 0.03])
        >>> compute_log_abf(beta=beta, se=se, se_prior=0.20)

    """

    # Shrinkage factor: ratio of the prior variance to the total variance
    r = (se_prior**2) / (se_prior**2 + se**2)

    # Z statistic
    z = beta / se

    # Log ABF
    log_abf = 0.5 * (np.log(1 - r) + (r * z**2))

    return log_abf
