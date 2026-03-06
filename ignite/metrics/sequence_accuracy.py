from collections.abc import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["SequenceAccuracy"]


class SequenceAccuracy(Metric):
    r"""Calculates the exact-match sequence accuracy for sequence-to-sequence models.

    This metric is designed for models whose output is of shape
    ``(batch_size, sequence_length, num_classes)`` and target of shape
    ``(batch_size, sequence_length)``, which is common in NLP tasks such as
    language modeling, machine translation, and text generation.

    A prediction is considered correct only if **every token** in the sequence
    matches the target (exact match). Padding tokens can be optionally excluded
    from the comparison.

    .. math::
        \text{SequenceAccuracy} = \frac{1}{N}\sum_{i=1}^{N}
        \mathbf{1}\left[\hat{y}_i^{(t)} = y_i^{(t)} \;\forall\, t \in \mathcal{M}_i\right]

    where :math:`\mathcal{M}_i` is the set of non-padding token positions for the
    :math:`i`-th sequence.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` must be in the shape ``(batch_size, sequence_length, num_classes)``
      where ``num_classes`` is the vocabulary size.
    - ``y`` must be in the shape ``(batch_size, sequence_length)``.

    Args:
        pad_idx: Index of the padding token to ignore when comparing sequences.
            If ``None``, all tokens are included in the comparison. Default: ``None``.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output
            into the form expected by the metric. Default: ``lambda x: x``.
        device: specifies which device updates are accumulated on. By default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`,
        visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SequenceAccuracy(pad_idx=0)
            metric.attach(default_evaluator, "seq_acc")
            # y_pred: (batch=2, seq_len=3, num_classes=5)
            y_pred = torch.tensor([
                [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
            ], dtype=torch.float)
            # y: (batch=2, seq_len=3)
            y = torch.tensor([[1, 2, 0], [3, 4, 0]])
            state = default_evaluator.run([[y_pred, y]])
            print(state.metrics["seq_acc"])

        .. testoutput::

            1.0

    .. versionadded:: 0.6.0
    """

    _state_dict_all_req_keys = ("_num_correct", "_num_examples")

    def __init__(
        self,
        pad_idx: int | None = None,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
    ):
        if pad_idx is not None and not isinstance(pad_idx, int):
            raise TypeError(f"Argument pad_idx should be an integer or None, got {type(pad_idx)}.")
        self.pad_idx = pad_idx
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndim != 3:
            raise ValueError(f"y_pred must have shape (batch_size, sequence_length, num_classes), got {y_pred.shape}.")

        if y.ndim != 2:
            raise ValueError(f"y must have shape (batch_size, sequence_length), got {y.shape}.")

        if y_pred.shape[:2] != y.shape:
            raise ValueError(
                f"Batch size and sequence length of y_pred {y_pred.shape[:2]} " f"and y {y.shape} must match."
            )

        # (batch, seq_len)
        predicted_tokens = torch.argmax(y_pred, dim=-1)

        if self.pad_idx is not None:
            # Mask out padding positions — treat them as always correct
            pad_mask = y == self.pad_idx
            correct_tokens = (predicted_tokens == y) | pad_mask
        else:
            correct_tokens = predicted_tokens == y

        # A sequence is correct only if all (non-padded) tokens match
        correct_sequences = correct_tokens.all(dim=-1)

        self._num_correct += correct_sequences.sum().to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("SequenceAccuracy must have at least one example before it can be computed.")
        return self._num_correct.item() / self._num_examples
