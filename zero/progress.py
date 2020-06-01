import enum


class LossScore:
    def __init__(self, loss_fn):
        self._loss_fn = loss_fn

    def loss(self):
        return self._loss_fn()

    def score(self):
        return -self.loss()


class _Status(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class ProgressTracker:
    def __init__(self, patience, min_delta=0.0):
        self._patience = patience
        self._min_delta = float(min_delta)
        self._best_score = None
        self._status = _Status.NEUTRAL
        self._bad_counter = 0

    @property
    def best_score(self):
        return self._best_score

    @property
    def success(self):
        return self._status == _Status.SUCCESS

    @property
    def fail(self):
        return self._status == _Status.FAIL

    def _set_success(self, score):
        self._best_score = score
        self._status = _Status.SUCCESS
        self._bad_counter = 0

    def update(self, score):
        if self._best_score is None:
            self._set_success(score)
        elif score > self._best_score + self._min_delta:
            self._set_success(score)
        else:
            self._bad_counter += 1
            self._status = (
                _Status.FAIL
                if self._bad_counter > self._patience else
                _Status.NEUTRAL
            )

    def forget_bad_updates(self):
        self._bad_counter = 0
        self._status = _Status.NEUTRAL

    def reset(self):
        self.forget_bad_updates()
        self._best_score = None
