import pytest
import torch

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import SequenceAccuracy


def test_no_update():
    metric = SequenceAccuracy()
    with pytest.raises(NotComputableError, match=r"SequenceAccuracy must have at least one example"):
        metric.compute()


def test_invalid_pad_idx():
    with pytest.raises(TypeError, match=r"Argument pad_idx should be an integer or None"):
        SequenceAccuracy(pad_idx=1.5)


def test_wrong_y_pred_shape():
    metric = SequenceAccuracy()
    # y_pred must be 3D
    with pytest.raises(ValueError, match=r"y_pred must have shape"):
        metric.update((torch.rand(4, 10), torch.randint(0, 10, (4,))))


def test_wrong_y_shape():
    metric = SequenceAccuracy()
    # y must be 2D
    with pytest.raises(ValueError, match=r"y must have shape"):
        metric.update((torch.rand(4, 5, 10), torch.randint(0, 10, (4,))))


def test_shape_mismatch():
    metric = SequenceAccuracy()
    # batch/seq mismatch between y_pred and y
    with pytest.raises(ValueError, match=r"Batch size and sequence length"):
        metric.update((torch.rand(4, 5, 10), torch.randint(0, 10, (4, 6))))


def test_perfect_accuracy_no_padding():
    """All tokens correct, no padding — expect 1.0."""
    metric = SequenceAccuracy()
    metric.reset()

    # batch=3, seq_len=4, num_classes=5
    y = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3], [4, 3, 2, 1]])
    y_pred = torch.zeros(3, 4, 5)
    for i in range(3):
        for j in range(4):
            y_pred[i, j, y[i, j]] = 1.0

    metric.update((y_pred, y))
    assert metric.compute() == pytest.approx(1.0)


def test_zero_accuracy_no_padding():
    """All tokens wrong — expect 0.0."""
    metric = SequenceAccuracy()
    metric.reset()

    y = torch.tensor([[1, 2, 3], [0, 1, 2]])
    y_pred = torch.zeros(2, 3, 5)
    # Put max mass on wrong class (class 0 when y != 0, class 1 when y == 0)
    for i in range(2):
        for j in range(3):
            wrong = (y[i, j].item() + 1) % 5
            y_pred[i, j, wrong] = 1.0

    metric.update((y_pred, y))
    assert metric.compute() == pytest.approx(0.0)


def test_partial_accuracy():
    """First sequence all correct, second has one wrong token — expect 0.5."""
    metric = SequenceAccuracy()
    metric.reset()

    y = torch.tensor([[1, 2], [3, 4]])
    y_pred = torch.zeros(2, 2, 5)
    # Sequence 0: correct
    y_pred[0, 0, 1] = 1.0
    y_pred[0, 1, 2] = 1.0
    # Sequence 1: first token wrong
    y_pred[1, 0, 0] = 1.0  # wrong (should be 3)
    y_pred[1, 1, 4] = 1.0  # correct

    metric.update((y_pred, y))
    assert metric.compute() == pytest.approx(0.5)


def test_with_padding():
    """Padding tokens (pad_idx=0) should be ignored in the comparison."""
    metric = SequenceAccuracy(pad_idx=0)
    metric.reset()

    # Sequence 0: tokens [1, 2, 0, 0] — pad positions 2,3 ignored
    # Sequence 1: tokens [3, 4, 0, 0] — pad positions 2,3 ignored
    y = torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]])
    y_pred = torch.zeros(2, 4, 5)
    # Both sequences fully correct on non-pad tokens
    y_pred[0, 0, 1] = 1.0
    y_pred[0, 1, 2] = 1.0
    y_pred[0, 2, 0] = 1.0  # pad — doesn't matter
    y_pred[0, 3, 0] = 1.0  # pad — doesn't matter
    y_pred[1, 0, 3] = 1.0
    y_pred[1, 1, 4] = 1.0
    y_pred[1, 2, 0] = 1.0  # pad
    y_pred[1, 3, 0] = 1.0  # pad

    metric.update((y_pred, y))
    assert metric.compute() == pytest.approx(1.0)


def test_all_padding_tokens_correct():
    """Sequence where a wrong prediction falls only on a pad token — still counts as correct."""
    metric = SequenceAccuracy(pad_idx=0)
    metric.reset()

    y = torch.tensor([[1, 0]])  # token 1 is real, token 0 is padding
    y_pred = torch.zeros(1, 2, 5)
    y_pred[0, 0, 1] = 1.0  # correct
    y_pred[0, 1, 3] = 1.0  # wrong prediction at pad pos — should be ignored

    metric.update((y_pred, y))
    assert metric.compute() == pytest.approx(1.0)


def test_multiple_updates():
    """Metric accumulates correctly across multiple batches."""
    metric = SequenceAccuracy()
    metric.reset()

    # Batch 1: 1 correct out of 2
    y1 = torch.tensor([[1, 2], [3, 4]])
    y_pred1 = torch.zeros(2, 2, 5)
    y_pred1[0, 0, 1] = 1.0
    y_pred1[0, 1, 2] = 1.0  # seq0 correct
    y_pred1[1, 0, 0] = 1.0  # seq1 wrong
    y_pred1[1, 1, 4] = 1.0
    metric.update((y_pred1, y1))

    # Batch 2: 1 correct out of 1
    y2 = torch.tensor([[2, 3]])
    y_pred2 = torch.zeros(1, 2, 5)
    y_pred2[0, 0, 2] = 1.0
    y_pred2[0, 1, 3] = 1.0  # seq correct
    metric.update((y_pred2, y2))

    # Total: 2 correct out of 3
    assert metric.compute() == pytest.approx(2 / 3)


def test_integration_with_engine():
    """SequenceAccuracy works correctly when attached to an Engine."""
    y = torch.tensor([[1, 2, 3], [0, 1, 2]])
    y_pred = torch.zeros(2, 3, 5)
    for i in range(2):
        for j in range(3):
            y_pred[i, j, y[i, j]] = 1.0  # perfect predictions

    def update_fn(engine, batch):
        return batch

    evaluator = Engine(update_fn)
    metric = SequenceAccuracy(pad_idx=0)
    metric.attach(evaluator, "seq_acc")

    state = evaluator.run([(y_pred, y)])
    assert state.metrics["seq_acc"] == pytest.approx(1.0)


def test_reset_clears_state():
    metric = SequenceAccuracy()
    y = torch.tensor([[1, 2]])
    y_pred = torch.zeros(1, 2, 5)
    y_pred[0, 0, 1] = 1.0
    y_pred[0, 1, 2] = 1.0

    metric.update((y_pred, y))
    assert metric._num_examples == 1

    metric.reset()
    assert metric._num_examples == 0
    with pytest.raises(NotComputableError):
        metric.compute()
