from tensor import Tensor
from typing import Any
from enum import Enum
import numpy as np


# np.random.seed(1)


class Parameter:
    def __init__(self, value: Tensor):
        self.value = value

    def update(self, value: Tensor):
        # Update the value without changing the computation graph
        self.value = value

class Initialization(Enum):
    XAVIER = 1
    HE = 2


class ActivationType(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3
    LINEAR = 4


def make_tensor(shape, initialization, **kwargs):
    if initialization == Initialization.XAVIER:
        n_inputs = kwargs['n_inputs']
        n_outputs = kwargs['n_outputs']
        x = np.sqrt(6 / (n_inputs + n_outputs))
        data = np.random.uniform(-x, x, shape)
    elif initialization == Initialization.HE:
        n_inputs = kwargs['n_inputs']
        std = np.sqrt(2 / n_inputs)
        data = np.random.randn(*shape) * std
    else:
        data = 2 * np.random.random(shape) - 1  # [-1, 1]
    return Tensor(data)


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def forward(self, X: Tensor) -> Tensor:
        pass

    @property
    def parameters(self):
        # Full list of module parameters
        parameters = list(self._parameters.values())
        for module in self._modules.values():
            parameters += module.parameters
        return parameters

    # Using custom __setattr__ and __getattr__ to enable PyTorch-like layers initialization syntax
    # Parameters are automatically added to the
    def __setattr__(self, key: str, val) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)


class Linear(Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        initialization: Initialization=Initialization.XAVIER,
        initialization_keywords=None
    ):
        super().__init__()
        if not initialization_keywords:
            initialization_keywords = {}
        weights = make_tensor((in_size, out_size, ), initialization, **initialization_keywords)
        self.weights = Parameter(weights)
        bias = make_tensor((out_size, ), initialization, **initialization_keywords)
        self.bias = Parameter(bias)

    def forward(self, X: Tensor) -> Tensor:
        # X is a Tensor in the shape (batch_size, in_size)
        # out = X @ self.weights + self.bias

        return X @ self.weights.value + self.bias.value


class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.relu()


class Sigmoid(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.sigmoid()


class Tanh(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.tanh()
