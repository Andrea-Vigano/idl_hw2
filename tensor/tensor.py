from typing import Optional, Iterable
from .past import Past
from .autodiff import backpropagate
from .functions import Add, MatMul, Mul, Inv, Square, Log, Exp, Neg, Sum, ReLU, Sigmoid, Tanh
import numpy as np


# Use this as instance hash
_tensor_count = 0


class Tensor:
    def __init__(self, data: np.ndarray, past: Optional[Past]=None, requires_gradient: bool=True):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count

        self.data = data
        self.past = past
        self.requires_gradient = requires_gradient
        self.gradient = None

    def accumulate_gradient(self, gradient):
        if not self.gradient:
            self.gradient = Tensor(np.zeros(self.data.shape))
        self.gradient += gradient

    @property
    def shape(self):
        return self.data.shape

    def __str__(self):
        return self.data.__str__()

    def __add__(self, other):
        return Add.apply(self, other)

    def __sub__(self, other):
        return Add.apply(self, -other)

    def __matmul__(self, other):
        return MatMul.apply(self, other)

    def __neg__(self):
        return Neg.apply(self)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __truediv__(self, other):
        return Mul.apply(self, Inv.apply(other))

    def square(self):
        return Square.apply(self)

    def log(self):
        return Log.apply(self)

    def exp(self):
        return Exp.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def tanh(self):
        return Tanh.apply(self)

    def sum(self, dim: int=None):
        if dim is None:
            return Sum.apply(self, Tensor(np.array([-1]), requires_gradient=False))
        else:
            return Sum.apply(self, Tensor(np.array([dim]), requires_gradient=False))

    def mean(self):
        # No dimension
        return self.sum() / Tensor(np.array([self.data.size]), requires_gradient=False)

    @property
    def parents(self):
        # Tree-like property
        assert self.past is not None
        return self.past.inputs

    @property
    def is_leaf(self):
        return not self.past and self.requires_gradient

    @property
    def is_constant(self):
        return not self.past and not self.requires_gradient

    def chain_rule(self, output_gradient) -> Iterable[tuple]:
        # Compute gradients for the inputs that generated this tensor
        assert self.past
        assert self.past.inputs
        assert self.past.function

        inputs_gradients = self.past.function._backward_proxy(output_gradient, self.past.cache)
        # Check dimensions
        assert len(inputs_gradients) == len(self.past.inputs)
        return zip(self.past.inputs, inputs_gradients)

    def backward(self, output_gradient=None):
        if not output_gradient:
            assert self.data.shape == (1, )
            output_gradient = Tensor(np.array([1.0]))
        # Apply backpropagation and compute gradients for the rest of the computational graph
        backpropagate(self, output_gradient)
