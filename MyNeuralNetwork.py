import numpy as np
from FullyConnectedLayer import FullyConnectedLayer
from typing import Sequence


class MyNeuralNetwork:
    def __init__(self, layers: Sequence[int], activation_functions: Sequence[str] = ()):
        self.layers = []
        # Not including the input layer as a full "layer"
        for i in range(0, len(layers) - 1):
            self.layers.append(FullyConnectedLayer(layers[i], layers[i + 1],
                                                   activation_functions[i] if activation_functions else 'relu'))

    def forward(self, input: np.array):
        cur_res = input
        for layer in self.layers:
            cur_res = layer.forward(cur_res)
        return cur_res


# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)

print(nn.forward(np.array([[1], [2], [3], [4]])))