from typing import Sequence
from module import Parameter
from tensor import Tensor
import numpy as np


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.value.gradient = None

    def step(self):
        pass


class GD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], learning_rate: float):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            # Use numpy
            # Compute mean of gradients wrt first dimension
            gradients_mean = np.sum(parameter.value.gradient.data, axis=0)
            new_value = parameter.value.data - self.learning_rate * gradients_mean
            parameter.update(Tensor(new_value))


class MomentumGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], learning_rate: float, momentum: float=0.9):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._states = {}
        for parameter in self.parameters:
            # Init state
            self._states[id(parameter)] = 0.0

    def step(self):
        for parameter in self.parameters:
            # Use numpy
            # Compute mean of gradients wrt first dimension
            velocity = self._states[id(parameter)]

            gradients_mean = np.sum(parameter.value.gradient.data, axis=0)

            momentum_gradient = self.momentum * velocity - self.learning_rate * gradients_mean

            new_value = parameter.value.data + momentum_gradient
            # Update state for parameter
            self._states[id(parameter)] = momentum_gradient
            parameter.update(Tensor(new_value))


class Adam(Optimizer):
    STEP = 'step'
    FIRST_ORDER = 'first_order'
    SECOND_ORDER = 'second_order'

    def __init__(
            self,
            parameters: Sequence[Parameter],
            learning_rate: float,
            beta1: float=0.9,
            beta2: float=0.999,
            eps: float=1e-8
    ):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._states = {}
        for parameter in self.parameters:
            # Init state
            self._states[id(parameter)] = {}

    def step(self):
        for parameter in self.parameters:
            # Use numpy
            # Compute mean of gradients wrt first dimension
            gradients_mean = np.sum(parameter.value.gradient.data, axis=0)
            state = self._states[id(parameter)]
            np_gradient = np.sum(parameter.value.gradient.data, axis=0)

            if len(state) == 0:
                # Init state
                state[self.STEP] = 0
                state[self.FIRST_ORDER] = np.zeros(np_gradient.shape)
                state[self.SECOND_ORDER] = np.zeros(np_gradient.shape)

            state[self.STEP] += 1
            state[self.FIRST_ORDER] = state[self.FIRST_ORDER] * self.beta1 + (1 - self.beta1) * np_gradient
            state[self.SECOND_ORDER] = state[self.SECOND_ORDER] * self.beta2 + (1 - self.beta2) * np_gradient ** 2

            bias_1 = 1. - self.beta1 ** state[self.STEP]
            bias_2 = 1. - self.beta2 ** state[self.STEP]

            step_size = self.learning_rate * np.sqrt(bias_2) / (bias_1)
            update = state[self.FIRST_ORDER] / (state[self.SECOND_ORDER] ** 0.5 + self.eps)

            new_value = parameter.value.data - step_size * update
            parameter.update(Tensor(new_value))
