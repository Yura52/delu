__all__ = ['Eval']

from typing import ClassVar, List, Optional

import torch


class _ModelsContext:
    # TODO (docs): takes effect only within `with`, not when constructed!
    _train: ClassVar[bool] = NotImplemented
    _grad: ClassVar[bool] = NotImplemented

    def __init__(self, *models: torch.nn.Module) -> None:
        assert models
        self._models = models
        self._training: Optional[List[bool]] = None
        self._grad_context = torch.enable_grad() if self._grad else torch.no_grad()

    def __enter__(self) -> None:
        self._training = []
        for x in self._models:
            self._training.append(x.training)
            x.train(self._train)
        self._grad_context.__enter__()  # type: ignore

    def __exit__(self, *args) -> bool:  # type: ignore
        for model, train in zip(self._models, self._training):  # type: ignore
            model.train(train)
        self._grad_context.__exit__(*args)  # type: ignore
        return False


class Eval(_ModelsContext):
    _train = False
    _grad = False
