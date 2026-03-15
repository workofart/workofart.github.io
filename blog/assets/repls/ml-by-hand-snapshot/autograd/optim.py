from collections.abc import Callable
from typing import Any

from autograd.tensor import Tensor


class Optimizer:
    def __init__(self, model_parameters: dict[str, Tensor], lr: float, **kwargs: Any) -> None:
        self.model_parameters = model_parameters
        self.lr = lr
        self._hyperparams = dict(kwargs)

    def _recursive_param_op(self, params: Any, update_fn: Callable[[Any], None]) -> None:
        if isinstance(params, dict):
            for value in params.values():
                self._recursive_param_op(value, update_fn)
        elif isinstance(params, (list, tuple)):
            for value in params:
                self._recursive_param_op(value, update_fn)
        elif hasattr(params, "grad"):
            update_fn(params)

    def zero_grad(self) -> None:
        def update_fn(param: Any) -> None:
            param.grad = None

        self._recursive_param_op(self.model_parameters, update_fn)

    def step(self) -> None:
        return None


class SGD(Optimizer):
    def __init__(self, model_parameters: dict[str, Tensor], lr: float, **kwargs: Any) -> None:
        super().__init__(model_parameters, lr=lr, **kwargs)

    def step(self) -> None:
        def update_fn(param: Any) -> None:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data

        self._recursive_param_op(self.model_parameters, update_fn)
