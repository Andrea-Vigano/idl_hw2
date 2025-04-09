from module import ActivationType, Initialization


NUMBER_OF_LAYERS = 'number_of_layers'
LAYERS_IO = 'layers_io'
LAYERS_ACTIVATIONS = 'layer_activations'
LAYERS_INITIALIZATIONS = 'layer_initializations'
INITIALIZATION_KEYWORDS = 'initialization_keywords'

default_mlp_config = {
    NUMBER_OF_LAYERS: 2,
    LAYERS_IO: ((3, 3), (3, 1)),
    LAYERS_ACTIVATIONS: (ActivationType.TANH, ActivationType.SIGMOID),
    LAYERS_INITIALIZATIONS: (Initialization.HE, Initialization.HE),
    INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
}

# Deliverable 2 configs
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 2,
#     LAYERS_IO: ((2, 1), (1, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.LINEAR),
#     LAYERS_INITIALIZATIONS: (Initialization.XAVIER, Initialization.XAVIER),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }

# Deliverable 3 configs
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 2,
#     LAYERS_IO: ((2, 64), (64, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.SIGMOID),
#     LAYERS_INITIALIZATIONS: (Initialization.XAVIER, Initialization.XAVIER),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }

# Deliverable 4 regressor config
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 2,
#     LAYERS_IO: ((2, 32), (32, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.LINEAR),
#     LAYERS_INITIALIZATIONS: (Initialization.XAVIER, Initialization.XAVIER),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }
#
# Deliverable 4 classifier config
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 2,
#     LAYERS_IO: ((2, 32), (32, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.SIGMOID),
#     LAYERS_INITIALIZATIONS: (Initialization.XAVIER, Initialization.XAVIER),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }

# Deliverable 5 config
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 3,
#     LAYERS_IO: ((2, 64), (64, 32), (32, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.RELU, ActivationType.SIGMOID),
#     LAYERS_INITIALIZATIONS: (Initialization.XAVIER, Initialization.XAVIER, Initialization.XAVIER),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }

# Deliverable 6 config
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 5,
#     LAYERS_IO: ((2, 32), (32, 64), (64, 16), (16, 4), (4, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.RELU, ActivationType.RELU, ActivationType.RELU, ActivationType.SIGMOID),
#     LAYERS_INITIALIZATIONS: (Initialization.HE, Initialization.HE, Initialization.HE, Initialization.HE, Initialization.HE),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }

# Deliverable 7 config - XOR
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 2,
#     LAYERS_IO: ((3, 1), (1, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.RELU, ActivationType.SIGMOID),
#     LAYERS_INITIALIZATIONS: (Initialization.HE, Initialization.HE),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }

# Deliverable 7 config - swiss-roll
# default_mlp_config = {
#     NUMBER_OF_LAYERS: 2,
#     LAYERS_IO: ((3, 3), (3, 1)),
#     LAYERS_ACTIVATIONS: (ActivationType.TANH, ActivationType.SIGMOID),
#     LAYERS_INITIALIZATIONS: (Initialization.HE, Initialization.HE),
#     INITIALIZATION_KEYWORDS: {'n_inputs': 2, 'n_outputs': 1}
# }
