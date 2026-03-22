"""
Tests for qhchina.analytics.cython_ext.statistics module.
Tests batch statistical functions (chi2, log-likelihood, Fisher's exact).
"""
import pytest
import numpy as np


class TestBatchChi2:
    """Tests for batch_chi2 in statistics.pyx."""

    def test_basic_chi2(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([10, 20], dtype=np.int64)
        b = np.array([5, 10], dtype=np.int64)
        c = np.array([3, 8], dtype=np.int64)
        d = np.array([20, 40], dtype=np.int64)
        stats, pvals = batch_chi2(a, b, c, d)
        assert stats.shape == (2,)
        assert pvals.shape == (2,)
        assert np.all(stats >= 0)
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_chi2_with_correction(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([10], dtype=np.int64)
        b = np.array([5], dtype=np.int64)
        c = np.array([3], dtype=np.int64)
        d = np.array([20], dtype=np.int64)
        stats_no, pvals_no = batch_chi2(a, b, c, d, correction=False)
        stats_yes, pvals_yes = batch_chi2(a, b, c, d, correction=True)
        assert stats_no[0] >= stats_yes[0]
        assert pvals_no[0] <= pvals_yes[0]

    def test_chi2_zero_expected_gives_nan(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([0], dtype=np.int64)
        b = np.array([0], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        d = np.array([0], dtype=np.int64)
        stats, pvals = batch_chi2(a, b, c, d)
        assert np.isnan(stats[0])
        assert np.isnan(pvals[0])

    def test_chi2_length_mismatch_raises(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([10, 20], dtype=np.int64)
        b = np.array([5], dtype=np.int64)
        c = np.array([3], dtype=np.int64)
        d = np.array([20], dtype=np.int64)
        with pytest.raises(ValueError, match="same length"):
            batch_chi2(a, b, c, d)


class TestBatchLogLikelihood:
    """Tests for batch_log_likelihood in statistics.pyx."""

    def test_basic_g2(self):
        from qhchina.analytics.cython_ext.statistics import batch_log_likelihood
        a = np.array([10], dtype=np.int64)
        b = np.array([5], dtype=np.int64)
        c = np.array([3], dtype=np.int64)
        d = np.array([20], dtype=np.int64)
        stats, pvals = batch_log_likelihood(a, b, c, d)
        assert stats.shape == (1,)
        assert pvals.shape == (1,)
        assert stats[0] > 0
        assert 0 <= pvals[0] <= 1

    def test_g2_equal_distributions(self):
        from qhchina.analytics.cython_ext.statistics import batch_log_likelihood
        a = np.array([10], dtype=np.int64)
        b = np.array([10], dtype=np.int64)
        c = np.array([10], dtype=np.int64)
        d = np.array([10], dtype=np.int64)
        stats, pvals = batch_log_likelihood(a, b, c, d)
        assert stats[0] == pytest.approx(0.0, abs=1e-10)
        assert pvals[0] == pytest.approx(1.0, abs=0.01)

    def test_g2_multiple_tables(self):
        from qhchina.analytics.cython_ext.statistics import batch_log_likelihood
        a = np.array([10, 50, 5], dtype=np.int64)
        b = np.array([5, 10, 100], dtype=np.int64)
        c = np.array([3, 20, 8], dtype=np.int64)
        d = np.array([20, 100, 200], dtype=np.int64)
        stats, pvals = batch_log_likelihood(a, b, c, d)
        assert len(stats) == 3
        assert len(pvals) == 3


class TestBatchFisherExact:
    """Verify batch_fisher_exact works from the statistics module."""

    def test_fisher_import_from_statistics(self):
        from qhchina.analytics.cython_ext.statistics import batch_fisher_exact
        a = np.array([1], dtype=np.int64)
        b = np.array([9], dtype=np.int64)
        c = np.array([11], dtype=np.int64)
        d = np.array([3], dtype=np.int64)
        pvals = batch_fisher_exact(a, b, c, d, 'two-sided')
        assert 0 <= pvals[0] <= 1

    def test_fisher_greater_alternative(self):
        from qhchina.analytics.cython_ext.statistics import batch_fisher_exact
        a = np.array([10], dtype=np.int64)
        b = np.array([2], dtype=np.int64)
        c = np.array([1], dtype=np.int64)
        d = np.array([10], dtype=np.int64)
        pvals = batch_fisher_exact(a, b, c, d, 'greater')
        assert 0 <= pvals[0] <= 1

    def test_fisher_less_alternative(self):
        from qhchina.analytics.cython_ext.statistics import batch_fisher_exact
        a = np.array([1], dtype=np.int64)
        b = np.array([10], dtype=np.int64)
        c = np.array([10], dtype=np.int64)
        d = np.array([2], dtype=np.int64)
        pvals = batch_fisher_exact(a, b, c, d, 'less')
        assert 0 <= pvals[0] <= 1
