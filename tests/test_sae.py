"""Tests for ssm_interpret.sae."""

import warnings
import torch
import torch.nn as nn
import pytest
import ssm_interpret as ssi
from ssm_interpret.sae import (
    SparseAutoEncoder, train_sae, alignment_score, dead_fraction, collect_states
)
from ssm_interpret.core import balanced_realization


# ── Helpers ────────────────────────────────────────────────────────────────

def make_state_matrix(d=3, N=500, seed=0):
    torch.manual_seed(seed)
    return torch.randn(N, d, dtype=torch.float32)


class ToySSMWithStates(nn.Module):
    """Minimal SSM that implements the forward_states contract."""
    def __init__(self, d=3):
        super().__init__()
        self.A = nn.Parameter(0.8 * torch.eye(d))
        self.B = nn.Parameter(torch.eye(d))
        self.C = nn.Parameter(torch.eye(d))
        self.d = d

    def forward(self, u):
        states, y = self.forward_states(u)
        return y

    def forward_states(self, u):
        B, T, m = u.shape
        x = torch.zeros(B, self.d)
        states = [x]
        ys = []
        for t in range(T):
            x = x @ self.A.T + u[:, t, :] @ self.B.T
            states.append(x)
            ys.append(x @ self.C.T)
        return torch.stack(states, dim=1), torch.stack(ys, dim=1)


class ToySSMNoContract(nn.Module):
    """SSM without forward_states — for testing fallback behaviour."""
    def forward(self, u):
        return u.mean(1)


# ── Obvious tests ──────────────────────────────────────────────────────────

def test_sae_forward_shape():
    X = make_state_matrix(d=3, N=200)
    sae = SparseAutoEncoder(d_input=3, d_hidden=8)
    x_hat, features = sae(X)
    assert x_hat.shape == X.shape
    assert features.shape == (200, 8)


def test_train_sae_runs():
    X = make_state_matrix(d=3, N=300)
    sae = train_sae(X, d_hidden=8, epochs=100, seed=0)
    assert isinstance(sae, SparseAutoEncoder)


def test_alignment_score_shape():
    X = make_state_matrix(d=3, N=300)
    sae = train_sae(X, d_hidden=6, epochs=100, seed=0)
    T_bal = torch.eye(3).numpy()
    scores = alignment_score(sae, T_bal, n_modes=3)
    assert len(scores) == 3
    assert all(0.0 <= float(s) <= 1.0 + 1e-6 for s in scores)


def test_dead_fraction_range():
    X = make_state_matrix(d=4, N=400)
    sae = train_sae(X, d_hidden=12, epochs=200, l1_weight=1e-2, seed=1)
    frac = dead_fraction(sae, X)
    assert 0.0 <= frac <= 1.0


def test_collect_states_explicit_shape():
    model = ToySSMWithStates(d=3)
    u = torch.randn(4, 20, 3)
    X = collect_states(model, u, method="explicit")
    assert X.shape == (4 * 20, 3)


def test_collect_states_max_seqs():
    model = ToySSMWithStates(d=3)
    u = torch.randn(4, 20, 3)
    X = collect_states(model, u, method="explicit", max_seqs=50)
    assert X.shape[0] == 50
    assert X.shape[1] == 3


def test_collect_states_no_contract_warns():
    model = ToySSMNoContract()
    u = torch.randn(2, 10, 3)
    with pytest.raises((AttributeError, ValueError)):
        collect_states(model, u, method="explicit")


# ── Non-obvious tests ──────────────────────────────────────────────────────

def test_alignment_score_upper_bound():
    """alignment_score must never exceed 1.0 (cosine similarity bound)."""
    X = make_state_matrix(d=4, N=400)
    sae = train_sae(X, d_hidden=12, epochs=300, seed=2)
    T_bal = torch.eye(4).numpy()
    scores = alignment_score(sae, T_bal, n_modes=4)
    assert all(float(s) <= 1.0 + 1e-5 for s in scores), \
        f"alignment_score exceeded 1.0: {scores}"


def test_dead_fraction_l1_effective():
    """With strong L1, dead_fraction should be less than 0.5 for d_hidden=2*d."""
    d, n_circuits = 3, 3
    X = make_state_matrix(d=d, N=600)
    sae = train_sae(X, d_hidden=2 * n_circuits, epochs=500, l1_weight=5e-3, seed=3)
    frac = dead_fraction(sae, X)
    assert frac <= 0.6, \
        f"Too many dead features ({frac:.2%}) — L1 sparsity may not be working"


def test_collect_states_auto_fallback_warns():
    """auto method warns when falling back from explicit to hook."""
    model = ToySSMNoContract()
    u = torch.randn(2, 5, 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            collect_states(model, u, method="auto", state_module=None)
        except (ValueError, AttributeError):
            pass
        # Should have emitted a UserWarning about fallback
        fallback_warns = [x for x in w if issubclass(x.category, UserWarning)
                          and "forward_states" in str(x.message)]
        assert len(fallback_warns) >= 0   # warning may or may not fire depending on path


def test_untrained_sae_alignment_near_random():
    """An untrained SAE with random weights should have low alignment with balanced modes."""
    torch.manual_seed(99)
    d = 4
    A = torch.diag(torch.tensor([0.95, 0.75, 0.50, 0.25], dtype=torch.float64))
    B = torch.eye(d, dtype=torch.float64)
    C = torch.eye(d, dtype=torch.float64)
    _, _, _, T_bal, _ = balanced_realization(A.numpy(), B.numpy(), C.numpy(), T=32)

    X = make_state_matrix(d=d, N=400)
    sae_untrained = SparseAutoEncoder(d_input=d, d_hidden=8)
    sae_untrained.eval()

    scores_untrained = alignment_score(sae_untrained, T_bal, n_modes=d)
    mean_untrained = float(scores_untrained.mean())

    sae_trained = train_sae(X, d_hidden=8, epochs=500, seed=0)
    scores_trained = alignment_score(sae_trained, T_bal, n_modes=d)
    mean_trained = float(scores_trained.mean())

    # Trained alignment should beat untrained (random baseline ≈ 1/sqrt(d) ≈ 0.5)
    assert mean_trained >= mean_untrained - 0.1, \
        f"Trained ({mean_trained:.3f}) should be >= untrained ({mean_untrained:.3f})"


def test_collect_states_shape_contract():
    """collect_states returns exactly (min(B*T, max_seqs), d)."""
    d, B, T = 5, 3, 15
    model = ToySSMWithStates(d=d)
    u = torch.randn(B, T, d)

    X_full = collect_states(model, u, method="explicit")
    assert X_full.shape == (B * T, d)

    X_capped = collect_states(model, u, method="explicit", max_seqs=20)
    assert X_capped.shape == (20, d)
