"""Tests for ssm_mech.surgery._python_scan -- CPU only, no model required."""

import numpy as np
import pytest
import torch

from ssm_mech.surgery import _python_scan


def _make_simple_ssm(batch=1, seq=10, d_inner=4, d_state=8, seed=0):
    rng    = np.random.default_rng(seed)
    u      = torch.tensor(rng.standard_normal((batch, seq, d_inner)),  dtype=torch.float32)
    dt     = torch.tensor(np.abs(rng.uniform(0.01, 0.1, (batch, seq, d_inner))), dtype=torch.float32)
    A_cont = torch.tensor(-np.abs(rng.uniform(0.5, 2.0, (d_inner, d_state))), dtype=torch.float32)
    B      = torch.tensor(rng.standard_normal((batch, seq, d_state)), dtype=torch.float32)
    C      = torch.tensor(rng.standard_normal((batch, seq, d_state)), dtype=torch.float32)
    D      = torch.tensor(rng.standard_normal(d_inner), dtype=torch.float32)
    return u, dt, A_cont, B, C, D


class TestPythonScan:
    def test_output_shape(self):
        u, dt, A_cont, B, C, D = _make_simple_ssm(batch=2, seq=12, d_inner=4, d_state=8)
        y = _python_scan(u, dt, A_cont, B, C, D)
        assert y.shape == (2, 12, 4)

    def test_no_surgery_is_deterministic(self):
        args = _make_simple_ssm()
        y1 = _python_scan(*args)
        y2 = _python_scan(*args)
        torch.testing.assert_close(y1, y2)

    def test_surgery_changes_output(self):
        u, dt, A_cont, B, C, D = _make_simple_ssm(d_inner=4, d_state=8)
        # k=1 rank projection (keep only mode-0)
        rng = np.random.default_rng(5)
        V   = torch.tensor(
            np.linalg.qr(rng.standard_normal((8, 8)))[0][:, :1][np.newaxis, :, :].repeat(4, axis=0),
            dtype=torch.float32,
        )  # (4, 8, 1)
        y_clean   = _python_scan(u, dt, A_cont, B, C, D, V_k=None)
        y_surgery = _python_scan(u, dt, A_cont, B, C, D, V_k=V)
        assert not torch.allclose(y_clean, y_surgery)

    def test_full_rank_surgery_is_noop(self):
        u, dt, A_cont, B, C, D = _make_simple_ssm(d_inner=4, d_state=8)
        # V = identity (d_state x d_state): rank-d_state projection is identity
        V_full = torch.eye(8).unsqueeze(0).expand(4, -1, -1)
        y_clean    = _python_scan(u, dt, A_cont, B, C, D, V_k=None)
        y_identity = _python_scan(u, dt, A_cont, B, C, D, V_k=V_full)
        torch.testing.assert_close(y_clean, y_identity, atol=1e-5, rtol=1e-4)

    def test_rank1_projection_is_rank_deficient(self):
        u, dt, A_cont, B, C, D = _make_simple_ssm(d_inner=4, d_state=8, seq=5)
        rng = np.random.default_rng(3)
        v1  = torch.tensor(
            np.linalg.qr(rng.standard_normal((8, 8)))[0][:, :1][np.newaxis, :, :].repeat(4, axis=0),
            dtype=torch.float32,
        )
        y = _python_scan(u, dt, A_cont, B, C, D, V_k=v1)
        # output must still have full seq coverage
        assert y.shape == u.shape
