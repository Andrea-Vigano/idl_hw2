from typing import Iterable, TYPE_CHECKING
import numpy as np
import tensor

if TYPE_CHECKING:
    from tensor import Tensor


def topological_sort(tensor: 'Tensor') -> Iterable['Tensor']:
    visited, order = set(), []

    def dfs(node: 'Tensor'):
        if node.unique_id in visited:
            return

        visited.add(node.unique_id)
        if not node.is_constant:
            if not node.is_leaf:
                for parent in node.parents:
                    dfs(parent)
            order.append(node)

    dfs(tensor)
    order.reverse()
    return order


def backpropagate(_input: 'Tensor', output_gradient: 'Tensor'):
    topological_order = topological_sort(_input)

    # Keep map of derivatives for non-leaf nodes
    derivatives = {node.unique_id: tensor.Tensor(np.array([0.0])) for node in topological_order}

    # Load first derivative
    derivatives[_input.unique_id] = output_gradient

    # Traverse the graph and propagate gradients
    for node in topological_order:
        current_gradient = derivatives[node.unique_id]

        # Load gradient only in leaf nodes
        if node.is_leaf:
            node.accumulate_gradient(current_gradient)
            continue

        # Load gradient in parents
        for parent, parent_derivative in node.chain_rule(current_gradient):
            if not parent.is_constant:
                derivatives[parent.unique_id] += parent_derivative
