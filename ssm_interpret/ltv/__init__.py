"""ssm_interpret.ltv — Experimental LTV / gated SSM extension.

.. warning::
   This submodule is **experimental**.  The API may change between minor
   versions.  Results are validated only for near-LTI regimes (GVI < 0.05).
   For genuinely selective models (GVI > 0.20), treat outputs as indicative.

Import example::

    from ssm_interpret.ltv import gramians_ltv, scope_report
"""

import warnings

from ssm_interpret.ltv.gramians import gramians_from_model, gramians_ltv
from ssm_interpret.ltv.scope import etmfd, gvi, llod, scope_report

warnings.warn(
    "ssm_interpret.ltv is experimental. "
    "API stability and validation scope are limited — see module docstring.",
    stacklevel=2,
    category=FutureWarning,
)

__all__ = [
    "gramians_ltv",
    "gramians_from_model",
    "etmfd",
    "llod",
    "gvi",
    "scope_report",
]
