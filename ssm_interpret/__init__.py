"""ssm-interpret: Control-theoretic mechanistic interpretability for SSMs.

Stable public API
-----------------
Core (Gramians, balanced realization, truncation):

    from ssm_interpret import (
        gramians, hsv, balanced_realization, truncate,
        impulse_response, glover_bound,
    )

Surgery (causal circuit ablation):

    from ssm_interpret import ablate, selectivity, faithfulness_ratio

Diagnostics (structure & scope):

    from ssm_interpret import (
        kalman_taxonomy, architecture_gap, sym_error, scope_check,
    )

SAE validation:

    from ssm_interpret import (
        SparseAutoEncoder, train_sae, alignment_score,
        dead_fraction, collect_states,
    )

Experimental LTV extension (gated / selective SSMs):

    from ssm_interpret.ltv import gramians_ltv, gramians_from_model
    from ssm_interpret.ltv import etmfd, llod, gvi, scope_report

Quick start
-----------
>>> import ssm_interpret as ssi
>>> Ab, Bb, Cb, T_bal, sigma = ssi.balanced_realization(A, B, C, T=64)
>>> A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
>>> ssi.scope_check(gvi_value=0.03)
{'regime': 'lti', 'applicable': True, ...}
"""

from ssm_interpret.core import (
    balanced_realization,
    glover_bound,
    gramians,
    hsv,
    impulse_response,
    truncate,
)
from ssm_interpret.diagnostics import (
    architecture_gap,
    kalman_taxonomy,
    scope_check,
    sym_error,
)
from ssm_interpret.sae import (
    SparseAutoEncoder,
    alignment_score,
    collect_states,
    dead_fraction,
    train_sae,
)
from ssm_interpret.surgery import ablate, faithfulness_ratio, selectivity

__version__ = "0.1.0"
__author__  = "SSM Circuits Project"
__license__ = "Apache-2.0"

__all__ = [
    # core
    "gramians", "hsv", "balanced_realization", "truncate",
    "impulse_response", "glover_bound",
    # surgery
    "ablate", "selectivity", "faithfulness_ratio",
    # diagnostics
    "kalman_taxonomy", "architecture_gap", "sym_error", "scope_check",
    # sae
    "SparseAutoEncoder", "train_sae", "alignment_score",
    "dead_fraction", "collect_states",
]
