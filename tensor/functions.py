import tensor
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensor import Tensor


# For numerical stability
EPS = 1e-6


class Function:
    @classmethod
    def _forward_proxy(cls, *inputs: 'Tensor', cache: list) -> 'Tensor':
        return cls.forward(*inputs, cache=cache)

    @classmethod
    def _backward_proxy(cls, output_gradient, cache: list) -> tuple['Tensor']:
        result = cls.backward(output_gradient, cache=cache)
        if isinstance(result, tuple):
            return result
        return (result, )

    @classmethod
    def apply(cls, *inputs: 'Tensor') -> 'Tensor':
        # Apply this function to given inputs, returns a new Tensor with Past if needed
        needs_grad = False
        for _input in inputs:
            if _input.requires_gradient:
                needs_grad = True
                break

        cache = []
        result = cls._forward_proxy(*inputs, cache=cache)

        if needs_grad:
            past = tensor.Past(function=cls, inputs=inputs, cache=cache)
            result.past = past
        return result

    # This interface is implemented by Function's children
    # @staticmethod
    # def forward(*inputs: Tensor, cache: tuple) -> Tensor:
    #     pass
    #
    # @staticmethod
    # def backward(output_gradient: Tensor, cache: tuple) -> tuple[Tensor]:
    #     pass


class Add(Function):
    @staticmethod
    def forward(a: 'Tensor', b: 'Tensor', cache: list) -> 'Tensor':
        result = a.data + b.data
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> tuple['Tensor', 'Tensor']:
        return output_gradient, output_gradient


class MatMul(Function):
    @staticmethod
    def forward(a: 'Tensor', b: 'Tensor', cache: list) -> 'Tensor':
        cache.append(a)
        cache.append(b)
        result = a.data @ b.data
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> tuple['Tensor', 'Tensor']:
        a, b = cache[0].data, cache[1].data
        output_gradient_value = output_gradient.data

        results = (output_gradient_value @ b.T, a.T @ output_gradient_value)
        return tensor.Tensor(results[0]), tensor.Tensor(results[1])


class Mul(Function):
    @staticmethod
    def forward(a: 'Tensor', b: 'Tensor', cache: list) -> 'Tensor':
        cache.append(a)
        cache.append(b)
        result = a.data * b.data
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> tuple['Tensor', 'Tensor']:
        a, b = cache[0].data, cache[1].data
        output_gradient_value = output_gradient.data

        results = (output_gradient_value * b, output_gradient_value * a)
        return tensor.Tensor(results[0]), tensor.Tensor(results[1])


class Inv(Function):
    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        cache.append(a)
        result = 1 / a.data
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = - (1.0 / a ** 2) * output_gradient.data
        return tensor.Tensor(result)


class Square(Function):
    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        cache.append(a)
        result = np.square(a.data)
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = output_gradient.data * 2 * a
        return tensor.Tensor(result)


class Log(Function):
    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        cache.append(a)
        assert np.all(a.data >= 0)
        result = np.log(a.data + EPS)
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = output_gradient.data / (a + EPS)
        return tensor.Tensor(result)


class Exp(Function):
    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        result = np.log(a.data)
        result_tensor = tensor.Tensor(result)
        cache.append(result_tensor)
        return result_tensor

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = output_gradient.data * a
        return tensor.Tensor(result)


class Neg(Function):
    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        result = -a.data
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        return tensor.Tensor(-output_gradient.data)


class Sum(Function):
    @staticmethod
    def forward(a: 'Tensor', dim: 'Tensor', cache: list) -> 'Tensor':
        # cache.append(a)
        # cache.append(dim)
        dimension = dim.data[0]
        if dimension == -1:
            result = np.array([np.sum(a.data)])
        else:
            result = np.array([np.sum(a.data, axis=dimension)])
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> tuple['Tensor', 'Tensor']:
        # a, dimension = cache[0].data, cache[1].data[0]
        results = (output_gradient.data, np.array([0.0]))
        return tensor.Tensor(results[0]), tensor.Tensor(results[1])


class ReLU(Function):
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        cache.append(a)
        result = ReLU.relu(a.data)
        return tensor.Tensor(result)

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = np.where(a > 0, output_gradient.data, 0.0)
        return tensor.Tensor(result)


class Sigmoid(Function):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        result = Sigmoid.sigmoid(a.data)
        result_tensor = tensor.Tensor(result)
        cache.append(result_tensor)
        return result_tensor

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = a * (-a + 1.0) * output_gradient.data
        return tensor.Tensor(result)


class Tanh(Function):
    @staticmethod
    def forward(a: 'Tensor', cache: list) -> 'Tensor':
        result = np.tanh(a.data)
        result_tensor = tensor.Tensor(result)
        cache.append(result_tensor)
        return result_tensor

    @staticmethod
    def backward(output_gradient: 'Tensor', cache: list) -> 'Tensor':
        a = cache[0].data
        result = output_gradient.data * (-(a ** 2) + 1)
        return tensor.Tensor(result)
