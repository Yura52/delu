"""Easier training process."""

__all__ = ['ProgressTracker', 'learn']

import enum
import math
import warnings
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import torch

T = TypeVar('T')


class _Status(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class ProgressTracker:
    """Tracks the best score, facilitates early stopping.

    For `~ProgressTracker`, **the greater score is the better score**.
    At any moment the tracker is in one of the following states:

    - success: the last update changed the best score
    - fail: last :code:`n > patience` updates are not better than the best score
    - neutral: if neither success nor fail

    Args:
        patience: Allowed number of bad updates. For example, if patience is 2, then
            2 bad updates is not a fail, but 3 bad updates is a fail. If `None`, then
            the progress tracker never fails.
        min_delta: minimal improvement over current best score to count it as success.

    Examples:
        .. testcode::

            progress = ProgressTracker(2)
            progress = ProgressTracker(3, 0.1)

    .. rubric:: Tutorial

    .. testcode::

        progress = ProgressTracker(2)
        progress.update(-999999999)
        assert progress.success  # the first update always updates the best score

        progress.update(123)
        assert progress.success
        assert progress.best_score == 123

        progress.update(0)
        assert not progress.success and not progress.fail

        progress.update(123)
        assert not progress.success and not progress.fail
        progress.update(123)
        # patience is 2 and the best score is not updated for more than 2 steps
        assert progress.fail
        assert progress.best_score == 123  # fail doesn't affect the best score
        progress.update(123)
        assert progress.fail  # still no improvements

        progress.forget_bad_updates()
        assert not progress.fail and not progress.success
        assert progress.best_score == 123
        progress.update(0)
        assert not progress.fail  # just 1 bad update (the patience is 2)

        progress.reset()
        assert not progress.fail and not progress.success
        assert progress.best_score is None
    """

    def __init__(self, patience: Optional[int], min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = float(min_delta)
        self._best_score: Optional[float] = None
        self._status = _Status.NEUTRAL
        self._bad_counter = 0

    @property
    def best_score(self) -> Optional[float]:
        """The best score so far.

        If the tracker is just created/reset, return `None`.
        """
        return self._best_score

    @property
    def success(self) -> bool:
        """Check if the tracker is in the 'success' state."""
        return self._status == _Status.SUCCESS

    @property
    def fail(self) -> bool:
        """Check if the tracker is in the 'fail' state."""
        return self._status == _Status.FAIL

    def _set_success(self, score: float) -> None:
        self._best_score = score
        self._status = _Status.SUCCESS
        self._bad_counter = 0

    def update(self, score: float) -> None:
        """Update the tracker's state.

        Args:
            score: the score to use for the update.
        """
        if self._best_score is None:
            self._set_success(score)
        elif score > self._best_score + self._min_delta:
            self._set_success(score)
        else:
            self._bad_counter += 1
            self._status = (
                _Status.FAIL
                if self._patience is not None and self._bad_counter > self._patience
                else _Status.NEUTRAL
            )

    def forget_bad_updates(self) -> None:
        """Reset bad updates and status, but not the best score."""
        self._bad_counter = 0
        self._status = _Status.NEUTRAL

    def reset(self) -> None:
        """Reset everything."""
        self.forget_bad_updates()
        self._best_score = None


def learn(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,  # type: ignore
    loss_fn: Callable[..., torch.Tensor],
    step: Callable[[T], Any],
    batch: T,
    star: bool = False,
) -> Tuple[float, Any]:
    """The "default" training step.

    The function does the following:

    #. Switches the model to the training mode, sets its gradients to zero.
    #. Performs the call :code:`step(batch)` or :code:`step(*batch)`
    #. The output from the previous step is passed to :code:`loss_fn`
    #. `torch.Tensor.backward` is applied to the obtained loss tensor.
    #. The optimization step is performed.
    #. Returns the loss's value (float) and :code:`step`'s output

    Args:
        model: the model to train
        optimizer: the optimizer for :code:`model`
        loss_fn: the function that takes :code:`step`'s output as input and returns a
            loss tensor
        step: the function that takes :code:`batch` as input and produces input for
            :code:`loss_fn`. Usually it is a function that applies the model to a batch
            and returns the result alogn with ground truth (if available). See
            examples below.
        batch: input for :code:`step`
        star: if True, then the output of :code:`step` is unpacked when passed to
            :code:`loss_fn`, i.e. :code:`loss_fn(*step_output)` is performed instead of
            :code:`loss_fn(step_output)`
    Returns:
        (loss_value, step_output)

    Note:
        After the function returns:

        - :code:`model`'s gradients (produced by backward) are **preserved**
        - :code:`model`'s state (training or not) is **undefined**

    Warning:
        If loss value is not finite (i.e. `math.isfinite` returns `False`), then
        backward and optimization step **are not performed** (you can still do it after
        the function returns, if needed). Additionally, `RuntimeWarning` is issued.

    Examples:

        .. code-block::

            model = ...
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()

            def step(batch):
                X, y = batch
                return model(X), y

            for batch in batches:
                learn(model, optimizer, loss_fn, step, batch, True)

        .. code-block::

            model = ...
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

            def step(batch):
                X, y = batch
                return {'y_pred': model(X), 'y': y}

            loss_fn = lambda out: torch.nn.functional.mse_loss(out['y_pred'], out['y'])

            for batch in batches:
                learn(model, optimizer, loss_fn, step, batch)
    """
    model.train()
    optimizer.zero_grad()
    out = step(batch)
    loss = loss_fn(*out) if star else loss_fn(out)
    loss_value = loss.item()
    if math.isfinite(loss_value):
        loss.backward()
        optimizer.step()
    else:
        warnings.warn(f'loss value is not finite: {loss_value}', RuntimeWarning)
    return cast(float, loss_value), out
