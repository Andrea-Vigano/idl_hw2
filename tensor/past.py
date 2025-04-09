from typing import Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .functions import Function
    from .tensor import Tensor


class Past:
    def __init__(self, function: Optional[Type['Function']], inputs: tuple['Tensor'], cache: list['Tensor']):
        self.function = function
        self.inputs = inputs
        self.cache = cache
