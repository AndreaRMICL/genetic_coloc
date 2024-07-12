import numpy as np

from genetic_coloc.coloc import coloc_abf


def test_coloc_not_colocalised():
    log_abf_trait_1 = np.array([1, 4, 9])
    log_abf_trait_2 = np.array([9, 4, 1])

    observed = coloc_abf(
        log_abf_trait_1=log_abf_trait_1, log_abf_trait_2=log_abf_trait_2
    )
    observed = np.round(observed, 2)
    expected = np.array([0.27, 0.22, 0.22, 0.18, 0.12])

    np.testing.assert_array_equal(observed, expected)


def test_coloc_colocalised():
    log_abf_trait_1 = np.array([1, 4, 9])
    log_abf_trait_2 = np.array([1, 4, 9])

    observed = coloc_abf(
        log_abf_trait_1=log_abf_trait_1, log_abf_trait_2=log_abf_trait_2
    )
    observed = np.round(observed, 2)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    np.testing.assert_array_equal(observed, expected)
