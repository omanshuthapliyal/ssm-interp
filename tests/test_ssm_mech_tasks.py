"""Tests for ssm_mech.tasks -- no GPU required."""

import numpy as np
import pytest

from ssm_mech.tasks import (
    IOIPrompt, InductionPrompt, SVAPrompt,
    build_ioi, build_induction, build_sva,
)


# ---------------------------------------------------------------------------
# IOI
# ---------------------------------------------------------------------------

class TestBuildIOI:
    def test_length(self):
        prompts = build_ioi(n=50)
        assert len(prompts) == 50

    def test_returns_ioi_prompt(self):
        p = build_ioi(n=1)[0]
        assert isinstance(p, IOIPrompt)
        assert p.text
        assert p.io_name
        assert p.subject_name

    def test_names_distinct(self):
        for p in build_ioi(n=30):
            assert p.io_name != p.subject_name

    def test_io_in_text(self):
        for p in build_ioi(n=20):
            assert p.io_name in p.text

    def test_subject_in_text(self):
        for p in build_ioi(n=20):
            assert p.subject_name in p.text

    def test_reproducible(self):
        a = build_ioi(n=10, seed=1)
        b = build_ioi(n=10, seed=1)
        assert all(pa.text == pb.text for pa, pb in zip(a, b))

    def test_different_seeds_differ(self):
        a = build_ioi(n=10, seed=1)
        b = build_ioi(n=10, seed=2)
        assert any(pa.text != pb.text for pa, pb in zip(a, b))


# ---------------------------------------------------------------------------
# Induction
# ---------------------------------------------------------------------------

class TestBuildInduction:
    def test_length(self):
        assert len(build_induction(n=40)) == 40

    def test_returns_induction_prompt(self):
        p = build_induction(n=1)[0]
        assert isinstance(p, InductionPrompt)
        assert p.text
        assert p.a_token
        assert p.b_token

    def test_kshot_pattern(self):
        for k in (1, 2, 3):
            p = build_induction(n=1, k_shot=k, seed=0)[0]
            a, b = p.a_token, p.b_token
            pattern = f" {a} {b}" * k + f" {a}"
            assert p.text == pattern

    def test_tokens_distinct(self):
        for p in build_induction(n=30):
            assert p.a_token != p.b_token

    def test_reproducible(self):
        a = build_induction(n=5, seed=7)
        b = build_induction(n=5, seed=7)
        assert all(pa.text == pb.text for pa, pb in zip(a, b))


# ---------------------------------------------------------------------------
# SVA
# ---------------------------------------------------------------------------

class TestBuildSVA:
    def test_length(self):
        assert len(build_sva(n=30)) == 30

    def test_returns_sva_prompt(self):
        p = build_sva(n=1)[0]
        assert isinstance(p, SVAPrompt)
        assert p.text
        assert p.correct in ("is", "are")
        assert p.incorrect in ("is", "are")

    def test_correct_incorrect_differ(self):
        for p in build_sva(n=20):
            assert p.correct != p.incorrect

    def test_mix_of_is_are(self):
        prompts = build_sva(n=100, seed=0)
        correct_verbs = {p.correct for p in prompts}
        assert "is" in correct_verbs
        assert "are" in correct_verbs

    def test_reproducible(self):
        a = build_sva(n=10, seed=5)
        b = build_sva(n=10, seed=5)
        assert all(pa.text == pb.text for pa, pb in zip(a, b))
