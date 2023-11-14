import numpy as np


class FullyConnectedLayer:
    """
    Class for storing a single fully connected layer together with the activation function
    """
    activation_functions = {
        'relu': {"forward": lambda x: np.maximum(x, 0), "derivative": lambda x: (x > 0) * 1},
        'linear': {"forward": lambda x: x, "derivative": lambda _: 1},
        'sigmoid': {"forward": lambda x:1/(1+np.exp(-x)), "derivative":lambda x: (1/(1+np.exp(-x))) * (1 - (1/(1+np.exp(-x)))) }
        # TODO: add more activation functions here. We can move the implementations to a separate file.
        #  Also think if this design is good enough, maybe wrap into some classes
    }

    def __init__(self, n_inputs: int, n_outputs: int, activation_function='relu'):
        if activation_function not in FullyConnectedLayer.activation_functions.keys():
            print(f"Error: activation function {activation_function} is not implemented")
            raise ValueError(f"{activation_function} is not implemented")
        self.h = 0
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
        self.h = self.w @ input - self.theta
        return FullyConnectedLayer.activation_functions[self.activation_function]["forward"](
            self.h)
    

    def backwards(self, deltaAbove: np.array ):


        derFunc = FullyConnectedLayer.activation_functions[self.activation_function]["derivative"]
        sum = 0
        for i in np.arange(0,len(self.w)):
            sum = sum + deltaAbove[i] * self.w[i]
        
        result = derFunc(self.h) * sum
        return result
    

    def backwardsfirst(self, resultOfforward, target)
        derFunc = FullyConnectedLayer.activation_functions[self.activation_function]["derivative"]
        return (derFunc(self.h) (resultOfforward - target))






