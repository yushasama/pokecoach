import torch

from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from deepmew_ex.experiments.v2.temporal_evaluation import score
from deepmew_ex.models.temporal import TemporalPolicy
from tests.datasets.v2.learning.test_admission import episode


def test_scoring_has_finite_baseline_and_does_not_change_checkpoint() -> None:
    torch.set_num_threads(1)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    sequence = actor_sequences(episode(0), 'validation')[0]
    with torch.no_grad():
        loss, baseline, count, ids = score(model, sequence)
    assert count == len(ids) > 0
    assert loss >= 0 and baseline >= 0
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
