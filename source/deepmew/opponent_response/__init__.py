"""Research-only opponent-response diagnostics.

Nothing in this package is imported by the submission or runtime path.
"""

from deepmew.opponent_response.baseline import run_response_ablation
from deepmew.opponent_response.prospective import run_prospective_response_ablation


__all__ = ["run_prospective_response_ablation", "run_response_ablation"]
