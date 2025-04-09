from tensor import Tensor
import numpy as np


def binary_cross_entropy(y: Tensor, labels: Tensor) -> Tensor:
    # y: (N, )  labels: (N, )
    # We suppose y contains probabilities of label being 1

    # Ones tensor
    ones = Tensor(np.ones(y.shape), requires_gradient=False)

    # Compute -((y.log() * labels).mean())
    return -(labels * (y.log()) + (ones - labels) * ((ones - y).log())).mean()


def l2(y: Tensor, labels: Tensor) -> Tensor:
    # y: (N, )  labels: (N, )
    # Compute (y - labels).square().mean()
    return (y - labels).square().mean()
