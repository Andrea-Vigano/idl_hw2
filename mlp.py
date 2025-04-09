from module import Module, ActivationType, Initialization, Linear, ReLU, Sigmoid, Tanh
from config import *
from tensor import Tensor


class MLP(Module):
    def __init__(self, config):
        super().__init__()
        self.number_of_layer = config[NUMBER_OF_LAYERS]
        self.layers = []

        # It is important to add the layers as attributes,
        # otherwise their parameters would not be included in the optimization
        for i in range(self.number_of_layer):
            # Add layer
            in_size, out_size = config[LAYERS_IO][i]
            initialization = config[LAYERS_INITIALIZATIONS][i]
            layer = Linear(in_size, out_size, initialization, config[INITIALIZATION_KEYWORDS])
            self.layers.append(layer)
            self.__setattr__(f'linear_layer_{i}', layer)
            activation = self._get_activation(config[LAYERS_ACTIVATIONS][i])
            if activation:
                self.layers.append(activation)
                self.__setattr__(f'activation_{i}', activation)

    @staticmethod
    def _get_activation(activation_type: ActivationType) -> Module:
        if activation_type == ActivationType.RELU:
            return ReLU()
        elif activation_type == ActivationType.SIGMOID:
            return Sigmoid()
        elif activation_type == ActivationType.TANH:
            return Tanh()
        else:
            # Linear activation means not activation
            return None

    def forward(self, X: Tensor) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out
