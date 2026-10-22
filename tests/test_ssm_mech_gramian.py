"""Tests for ssm_mech.gramian -- no GPU required."""

import numpy as np
import pytest

from ssm_mech.gramian import (
    OnlineStats,
    balanced_eigvec,
    gramian_from_stats,
    merge_stats,
)


# ---------------------------------------------------------------------------
# OnlineStats
# ---------------------------------------------------------------------------

class TestOnlineStats:
    def test_init_zeros(self):
        s = OnlineStats(d_inner=4, d_state=8)
        assert s.n == 0
        assert s.mean.shape == (4, 8)
        assert s.M2.shape   == (4, 8)

    def test_single_batch_mean(self):
        s = OnlineStats(d_inner=2, d_state=3)
        batch = np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 0.95]]])
        s.update(batch)
        assert s.n == 1
        np.testing.assert_allclose(s.A_avg, batch[0], rtol=1e-6)

    def test_welford_mean_matches_numpy(self):
        rng   = np.random.default_rng(0)
        batch = rng.uniform(0.1, 0.9, size=(100, 6, 4))
        s     = OnlineStats(d_inner=6, d_state=4)
        # Feed in chunks of varying size
        s.update(batch[:40])
        s.update(batch[40:75])
        s.update(batch[75:])
        np.testing.assert_allclose(s.A_avg, batch.mean(axis=0), atol=1e-10)

    def test_welford_variance_matches_numpy(self):
        rng   = np.random.default_rng(1)
        batch = rng.uniform(0.1, 0.9, size=(200, 4, 8))
        s     = OnlineStats(d_inner=4, d_state=8)
        s.update(batch)
        expected_var = batch.var(axis=0, ddof=1)
        computed_var = s.M2 / (s.n - 1)
        np.testing.assert_allclose(computed_var, expected_var, rtol=1e-8)

    def test_etmfd_lti_near_zero(self):
        # If A_disc is constant across tokens, ETMFD should be near 0
        s = OnlineStats(d_inner=4, d_state=8)
        A = 0.9 * np.ones((1, 4, 8))
        for _ in range(50):
            s.update(A)
        assert s.etmfd < 1e-5

    def test_etmfd_ltv_nonzero(self):
        rng = np.random.default_rng(42)
        s   = OnlineStats(d_inner=4, d_state=8)
        s.update(rng.uniform(0.2, 0.95, size=(500, 4, 8)))
        assert s.etmfd > 0.05

    def test_bc_accumulation(self):
        rng    = np.random.default_rng(7)
        s      = OnlineStats(d_inner=4, d_state=8)
        A_disc = rng.uniform(0.1, 0.9, size=(50, 4, 8))
        B      = rng.standard_normal((50, 8)).astype(np.float64)
        C      = rng.standard_normal((50, 8)).astype(np.float64)
        s.update(A_disc, B, C)
        np.testing.assert_allclose(s.B_avg, B.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(s.C_avg, C.mean(axis=0), rtol=1e-6)

    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(3)
        s   = OnlineStats(d_inner=4, d_state=8)
        s.update(rng.uniform(0.1, 0.9, size=(100, 4, 8)))
        path = tmp_path / "stats.npz"
        s.save(path)
        s2 = OnlineStats.load(path)
        np.testing.assert_array_equal(s.mean, s2.mean)
        np.testing.assert_array_equal(s.M2,   s2.M2)
        assert s.n == s2.n


# ---------------------------------------------------------------------------
# merge_stats
# ---------------------------------------------------------------------------

class TestMergeStats:
    def test_merge_equals_combined_update(self):
        rng   = np.random.default_rng(9)
        batch = rng.uniform(0.1, 0.9, size=(200, 4, 8))

        s_full = OnlineStats(d_inner=4, d_state=8)
        s_full.update(batch)

        s1 = OnlineStats(d_inner=4, d_state=8)
        s1.update(batch[:120])
        s2 = OnlineStats(d_inner=4, d_state=8)
        s2.update(batch[120:])
        s_merged = merge_stats(s1, s2)

        np.testing.assert_allclose(s_merged.A_avg, s_full.A_avg, atol=1e-10)
        np.testing.assert_allclose(s_merged.M2,    s_full.M2,    atol=1e-8)
        assert s_merged.n == s_full.n

    def test_merge_empty_left(self):
        rng = np.random.default_rng(11)
        s1  = OnlineStats(d_inner=4, d_state=8)    # empty
        s2  = OnlineStats(d_inner=4, d_state=8)
        s2.update(rng.uniform(0.1, 0.9, size=(50, 4, 8)))
        merged = merge_stats(s1, s2)
        assert merged.n == s2.n

    def test_merge_shape_mismatch_raises(self):
        s1 = OnlineStats(d_inner=4, d_state=8)
        s2 = OnlineStats(d_inner=8, d_state=8)
        with pytest.raises(ValueError):
            merge_stats(s1, s2)


# ---------------------------------------------------------------------------
# gramian_from_stats
# ---------------------------------------------------------------------------

class TestGramianFromStats:
    def test_symmetry(self):
        a = np.array([0.9, 0.7, 0.5])
        b = np.array([1.0, 0.5, 0.3])
        c = np.array([0.8, 0.6, 0.2])
        Wc, Wo = gramian_from_stats(a, b, c)
        np.testing.assert_allclose(Wc, Wc.T, atol=1e-10)
        np.testing.assert_allclose(Wo, Wo.T, atol=1e-10)

    def test_positive_semidefinite(self):
        a  = np.array([0.9, 0.7, 0.5, 0.3])
        b  = np.random.default_rng(0).uniform(-1, 1, 4)
        c  = np.random.default_rng(1).uniform(-1, 1, 4)
        Wc, _ = gramian_from_stats(a, b, c)
        eigvals = np.linalg.eigvalsh(Wc)
        assert eigvals.min() >= -1e-10


# ---------------------------------------------------------------------------
# balanced_eigvec
# ---------------------------------------------------------------------------

class TestBalancedEigvec:
    def _make_stats(self, d_inner=4, d_state=8, seed=0):
        rng    = np.random.default_rng(seed)
        s      = OnlineStats(d_inner=d_inner, d_state=d_state)
        A_disc = rng.uniform(0.2, 0.95, size=(200, d_inner, d_state))
        B      = rng.standard_normal((200, d_state))
        C      = rng.standard_normal((200, d_state))
        s.update(A_disc, B, C)
        return s

    def test_output_shape(self):
        s = self._make_stats(d_inner=4, d_state=8)
        V = balanced_eigvec(s)
        assert V.shape == (4, 8, 8)

    def test_columns_orthonormal(self):
        s = self._make_stats(d_inner=4, d_state=8)
        V = balanced_eigvec(s)
        for ch in range(4):
            G = V[ch].T @ V[ch]
            np.testing.assert_allclose(G, np.eye(8), atol=1e-5)

    def test_dtype_float32(self):
        s = self._make_stats()
        V = balanced_eigvec(s)
        assert V.dtype == np.float32
