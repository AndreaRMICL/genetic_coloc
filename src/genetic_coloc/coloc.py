from typing import NamedTuple

import numpy as np


class ColocProbabilities(NamedTuple):
    """Aggregates the COLOC probabilities for H0-H4.

    Args:
        pp_h0: No association with either trait.
        pp_h1: Association with trait 1, but not trait 2.
        pp_h2: Association with trait 2, but not trait 1.
        pp_h3: Assocation with trait 1 and 2, two independent SNPs.
        pp_h4: Assocation with trait 1 and 2, one shared SNP.
    """

    pp_h0: np.ndarray
    pp_h1: np.ndarray
    pp_h2: np.ndarray
    pp_h3: np.ndarray
    pp_h4: np.ndarray


def _logsum(values: np.ndarray) -> float:
    max_value = np.max(values)
    return max_value + np.log(np.sum(np.exp(values - max_value)))


def _logdiff(values_x: np.ndarray, values_y: np.ndarray) -> float:
    max_value = np.max((values_x, values_y))

    return max_value + np.log(
        np.exp(values_x - max_value) - np.exp(values_y - max_value)
    )


def coloc_abf(
    log_abf_trait_1: np.ndarray,
    log_abf_trait_2: np.ndarray,
    prior_trait_1: float = 1e-04,
    prior_trait_2: float = 1e-04,
    prior_coloc: float = 1e-05,
) -> ColocProbabilities:
    """Performs colocalisation analysis [1]_.

    Args:
        log_abf_trait_1: The log ABF values for trait 1.
        log_abf_trait_2: The log ABF values for trait 2. Should have the same SNPs (and
            the same order) as trait 1.
        prior_trait_1: The prior probability of a SNP being associated with trait 1.
        prior_trait_2: The prior probability of a SNP being associated with trait 2.
        prior_coloc: The prior probability of a SNP being associated with both traits.

    Returns:
        A NamedTuple containing the posterior probabilities for H0, H1, H2, H3, and H4.

    Raises:
        ValueError: If the lengths of `log_abf_trait_1` and `log_abf_trait_2` are not the same.

    Example:
        >>> import numpy as np
        >>> from genetic_coloc import coloc_abf
        >>> log_abf_trait_1 = np.array([1, 5, 10])
        >>> log_abf_trait_2 = np.array([1, 4, 9])
        >>> coloc_abf(log_abf_trait_1=log_abf_trait_1, log_abf_trait_2=log_abf_trait_2)

    References:
        [1] https://rdrr.io/cran/coloc/src/R/claudia.R

    """
    if len(log_abf_trait_1) != len(log_abf_trait_2):
        raise ValueError("Trait_1 and trait_2 should have the same length")

    log_sum = log_abf_trait_1 + log_abf_trait_2

    log_h0 = 0
    log_h1 = np.log(prior_trait_1) + _logsum(log_abf_trait_1)
    log_h2 = np.log(prior_trait_2) + _logsum(log_abf_trait_2)
    log_h3 = (
        np.log(prior_trait_1)
        + np.log(prior_trait_2)
        + _logdiff(
            _logsum(log_abf_trait_1) + _logsum(log_abf_trait_2), _logsum(log_sum)
        )
    )
    log_h4 = np.log(prior_coloc) + _logsum(log_sum)
    log_h_all = np.array([log_h0, log_h1, log_h2, log_h3, log_h4])

    log_denominator = _logsum(log_h_all)

    pp = np.exp(log_h_all - log_denominator)

    return ColocProbabilities(
        pp_h0=pp[0], pp_h1=pp[1], pp_h2=pp[2], pp_h3=pp[3], pp_h4=pp[4]
    )
