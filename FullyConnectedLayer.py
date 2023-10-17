import numpy as np


class FullyConnectedLayer:
    """
    Class for storing a single fully connected layer together with the activation function
    """
    activation_functions = {
        'relu': {"forward": lambda x: np.maximum(x, 0), "derivative": lambda x: (x > 0) * 1},
        'linear': {"forward": lambda x: x, "derivative": lambda _: 1}
        # TODO: add more activation functions here. We can move the implementations to a separate file.
        #  Also think if this design is good enough, maybe wrap into some classes
    }

    def __init__(self, n_inputs: int, n_outputs: int, activation_function='relu'):
        if activation_function not in FullyConnectedLayer.activation_functions.keys():
            print(f"Error: activation function {activation_function} is not implemented")
            raise ValueError(f"{activation_function} is not implemented")

        self.activation_function = activation_function
        # Depending on the activation function, weights are initialized in a different way.
        # In case of ReLu we use He initialization
        if activation_function == 'relu':
            self.w = np.random.randn(n_outputs, n_inputs).astype(np.float32) * np.sqrt(2 / n_inputs)
            self.theta = np.random.randn(n_outputs, 1) * np.sqrt(2 / n_inputs)

    def forward(self, input: np.array):
        """
        Perform the forward run, i.e. activation(W * input - theta)
        :param input: Input of the layer
        :return: Output of the layer after the activation function
        """
        return FullyConnectedLayer.activation_functions[self.activation_function]["forward"](
            self.w @ input - self.theta)
