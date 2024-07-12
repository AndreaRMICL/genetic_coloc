import numpy as np

from genetic_coloc.abf import compute_log_abf


def test_abf_binary_trait():
    beta = np.array([0.1, 0.2, 0.3])
    se = np.array([0.01, 0.02, 0.03])
    observed = np.round(compute_log_abf(beta=beta, se=se, se_prior=0.20), 2)
    expected = np.array([46.88, 47.20, 46.99])  # based on the R implementation

    np.testing.assert_array_equal(observed, expected)


def test_abf_quantitative_trait():
    beta = np.array([0.1, 0.2, 0.3])
    se = np.array([0.01, 0.02, 0.03])
    observed = np.round(compute_log_abf(beta=beta, se=se, se_prior=0.15), 2)
    expected = np.array([47.07, 47.10, 46.45])  # based on the R implementation

    np.testing.assert_array_equal(observed, expected)
