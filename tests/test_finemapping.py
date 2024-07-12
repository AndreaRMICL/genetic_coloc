import numpy as np

from genetic_coloc.finemapping import finemapping_abf


def test_finemapping_default_prior():
    log_abf = np.array([47.06849, 47.10292, 46.44787])
    prior = np.array([1e-04, 1e-04, 1e-04])

    expected = np.round(finemapping_abf(log_abf=log_abf, prior=prior), 2)
    observed = np.array([0.39, 0.40, 0.21])  # from the R implementation

    np.testing.assert_array_equal(expected, observed)


def test_finemapping_non_default_prior():
    log_abf = np.array([47.06849, 47.10292, 46.44787])
    prior = np.array([1e-4, 1e-4, 1e-02])

    expected = np.round(finemapping_abf(log_abf=log_abf, prior=prior), 2)
    observed = np.array([0.02, 0.02, 0.96])

    np.testing.assert_array_equal(expected, observed)
