"""
ssm_mech.tasks -- Built-in prompt sets and evaluators for behavioral probing.

Three tasks are provided:

  IOI (Indirect Object Identification)
      "When Mary and John went to the store, John gave a book to ___"
      Metric: logit difference IO_name - subject_name at the final token.

  Induction
      "cat dog cat dog cat dog cat ___"  (k-shot, predict next token after repeat)
      Metric: logit difference target_B - target_A at the final token.

  SVA (Subject-Verb Agreement)
      "The key to the cabinets ___"  (is vs are)
      Metric: log P(is) - log P(are) at the final token.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# IOI
# ---------------------------------------------------------------------------

_IOI_NAMES = [
    "Mary", "John", "Emma", "Michael", "Sarah", "David",
    "Jessica", "James", "Ashley", "Robert", "Jennifer", "William",
    "Amanda", "Richard", "Emily", "Daniel", "Stephanie", "Matthew",
]
_IOI_PLACES  = ["store", "park", "garden", "library", "restaurant", "office", "school"]
_IOI_OBJECTS = ["book", "apple", "gift", "letter", "box", "bag", "key"]
_IOI_TEMPLATES = [
    "When {io} and {s} went to the {place}, {s} gave a {obj} to",
    "Then {io} and {s} went to the {place}. {s} gave a {obj} to",
    "After {io} and {s} went to the {place}, {s} gave a {obj} to",
]


@dataclass
class IOIPrompt:
    text: str
    io_name: str
    subject_name: str


def build_ioi(
    n: int,
    seed: int = 42,
    names: Optional[List[str]] = None,
    places: Optional[List[str]] = None,
    objects: Optional[List[str]] = None,
) -> List[IOIPrompt]:
    """
    Generate n IOI prompts.

    Each prompt is a sentence where the indirect object (IO) and subject (S)
    are two distinct names. The correct completion is the IO name.

    Parameters
    ----------
    n : int
        Number of prompts to generate.
    seed : int
        Random seed for reproducibility.
    names, places, objects : lists, optional
        Override the default vocabulary.

    Returns
    -------
    List[IOIPrompt]
    """
    rng    = np.random.default_rng(seed)
    names  = list(names  or _IOI_NAMES)
    places = list(places or _IOI_PLACES)
    objects = list(objects or _IOI_OBJECTS)
    prompts = []
    for _ in range(n):
        name_pool = list(names)
        rng.shuffle(name_pool)
        io, s = name_pool[0], name_pool[1]
        tmpl  = _IOI_TEMPLATES[int(rng.integers(len(_IOI_TEMPLATES)))]
        text  = tmpl.format(
            io=io, s=s,
            place=places[int(rng.integers(len(places)))],
            obj=objects[int(rng.integers(len(objects)))],
        )
        prompts.append(IOIPrompt(text=text, io_name=io, subject_name=s))
    return prompts


@torch.no_grad()
def eval_ioi(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[IOIPrompt],
    device: torch.device,
) -> float:
    """
    Evaluate IOI logit difference (IO name minus subject name) averaged over prompts.

    Higher = model correctly predicts the indirect object.
    """
    diffs = []
    for p in prompts:
        ids    = tokenizer(p.text, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids).logits[0, -1, :]
        io_ids = tokenizer.encode(" " + p.io_name, add_special_tokens=False)
        s_ids  = tokenizer.encode(" " + p.subject_name, add_special_tokens=False)
        if io_ids and s_ids:
            diffs.append(float(logits[io_ids[0]] - logits[s_ids[0]]))
    return float(np.mean(diffs)) if diffs else float("nan")


# ---------------------------------------------------------------------------
# Induction
# ---------------------------------------------------------------------------

_IND_VOCAB = [
    "cat", "dog", "tree", "house", "book", "bird", "fish", "rain",
    "moon", "sun", "river", "stone", "cloud", "wind", "fire", "lake",
    "rose", "star", "leaf", "snow", "fox", "bear", "wolf", "dove",
    "hawk", "pine", "fern", "vine", "reed", "ash", "salt", "silk",
]


@dataclass
class InductionPrompt:
    text: str
    a_token: str    # first token in the repeating pair (a b) repeated k times + a
    b_token: str    # correct completion is b


def build_induction(
    n: int,
    k_shot: int = 2,
    seed: int = 42,
    vocab: Optional[List[str]] = None,
) -> List[InductionPrompt]:
    """
    Generate n induction prompts of the form: "a b a b ... a"
    where the correct next token is b.

    Parameters
    ----------
    n : int
    k_shot : int
        Number of (a, b) repetitions before the final a. Default 2.
    seed : int
    vocab : list of str, optional

    Returns
    -------
    List[InductionPrompt]
    """
    rng   = np.random.default_rng(seed)
    vocab = list(vocab or _IND_VOCAB)
    prompts = []
    for _ in range(n):
        v = list(vocab)
        rng.shuffle(v)
        a, b  = v[0], v[1]
        text  = f" {a} {b}" * k_shot + f" {a}"
        prompts.append(InductionPrompt(text=text, a_token=a, b_token=b))
    return prompts


@torch.no_grad()
def eval_induction(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[InductionPrompt],
    device: torch.device,
) -> float:
    """
    Evaluate induction logit difference (b_token minus a_token) averaged over prompts.

    Higher = model correctly predicts the repeated pattern.
    """
    diffs = []
    for p in prompts:
        ids    = tokenizer(p.text, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids).logits[0, -1, :]
        b_ids  = tokenizer.encode(" " + p.b_token, add_special_tokens=False)
        a_ids  = tokenizer.encode(" " + p.a_token, add_special_tokens=False)
        if b_ids and a_ids:
            diffs.append(float(logits[b_ids[0]] - logits[a_ids[0]]))
    return float(np.mean(diffs)) if diffs else float("nan")


# ---------------------------------------------------------------------------
# SVA (Subject-Verb Agreement)
# ---------------------------------------------------------------------------

_SVA_SINGULAR_TEMPLATES = [
    "The {noun} on the shelf",
    "The key to the {noun}",
    "The report about the {noun}",
    "The manager of the {noun}",
]
_SVA_PLURAL_TEMPLATES = [
    "The {noun} on the shelf",
    "The keys to the {noun}",
    "The reports about the {noun}",
    "The managers of the {noun}",
]
_SVA_SINGULAR_NOUNS = ["cabinet", "drawer", "room", "box", "case", "door"]
_SVA_PLURAL_NOUNS   = ["cabinets", "drawers", "rooms", "boxes", "cases", "doors"]


@dataclass
class SVAPrompt:
    text: str
    correct: str    # "is" or "are"
    incorrect: str


def build_sva(n: int, seed: int = 42) -> List[SVAPrompt]:
    """
    Generate n SVA prompts (mix of singular and plural attractors).

    Returns
    -------
    List[SVAPrompt]
    """
    rng = np.random.default_rng(seed)
    prompts = []
    for _ in range(n):
        if rng.integers(2) == 0:
            tmpl  = _SVA_SINGULAR_TEMPLATES[int(rng.integers(len(_SVA_SINGULAR_TEMPLATES)))]
            noun  = _SVA_SINGULAR_NOUNS[int(rng.integers(len(_SVA_SINGULAR_NOUNS)))]
            correct, incorrect = "is", "are"
        else:
            tmpl  = _SVA_PLURAL_TEMPLATES[int(rng.integers(len(_SVA_PLURAL_TEMPLATES)))]
            noun  = _SVA_PLURAL_NOUNS[int(rng.integers(len(_SVA_PLURAL_NOUNS)))]
            correct, incorrect = "are", "is"
        prompts.append(SVAPrompt(
            text=tmpl.format(noun=noun),
            correct=correct,
            incorrect=incorrect,
        ))
    return prompts


@torch.no_grad()
def eval_sva(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[SVAPrompt],
    device: torch.device,
) -> float:
    """
    Evaluate SVA logit difference (correct minus incorrect verb) averaged over prompts.
    """
    diffs = []
    for p in prompts:
        ids = tokenizer(p.text, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids).logits[0, -1, :]
        c_ids  = tokenizer.encode(" " + p.correct,   add_special_tokens=False)
        i_ids  = tokenizer.encode(" " + p.incorrect, add_special_tokens=False)
        if c_ids and i_ids:
            diffs.append(float(logits[c_ids[0]] - logits[i_ids[0]]))
    return float(np.mean(diffs)) if diffs else float("nan")
