import numpy as np


def finemapping_abf(log_abf: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Performs fine-mapping using Approximate Bayes Factor (ABF).

    Calculates the posterior probabilities of each SNP being causal
    using the Approximate Bayes Factor (ABF) approach [1]_.

    Args:
        log_abf: The log ABF values for each SNP.
        prior: The prior probabilities for each SNP being causal. The
            sum of the prior probabilities should be less than 1.

    Returns:
        The posterior probabilities for each SNP being causal.

    Raises:
        ValueError: If the sum of the prior probabilities is greater than or equal to 1.

    Example:
        >>> import numpy as np
        >>> from genetic_coloc import finemapping_abf
        >>> log_abf = np.array([46.87, 47.19, 46.99])
        >>> prior = np.array([1e-04, 1e-04, 1e-04])
        >>> finemapping_abf(log_abf=log_abf, prior=prior)

    References:
        [1] https://rdrr.io/cran/coloc/src/R/claudia.R
    """

    log_prior = np.log(prior)
    log_joint = log_abf + log_prior
    prior_sum = np.sum(prior)

    if prior_sum >= 1:
        raise ValueError("The sum of the priors needs to be < 1")

    log_not_causal = np.log(1 - prior_sum)
    log_joint_max = np.max(log_joint)

    log_denominator = log_joint_max + np.log(
        np.sum(np.exp(log_joint - log_joint_max))
        + np.exp(log_not_causal - log_joint_max)
    )

    return np.exp(log_joint - log_denominator)
