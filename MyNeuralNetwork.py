import numpy as np
from FullyConnectedLayer import FullyConnectedLayer
from typing import Sequence

# epoch: One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
# learningRate: 
class MyNeuralNetwork:
    def __init__(self, layers: Sequence[int], activation_functions: Sequence[str], epochNumber : int, learningrate, momentum = () ):
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
    
    def backward(self, vala):
      for i in range(0, len(layers) - 1):
       val = vala
       val = layer[i].backward(val)
      return val
    
    def fit(self,X,y):
        # train network with float feature vectores in x with target values of the samples values y
        o = self.forward(np.array([[1], [2], [3], [4]]))
        #delta = o- y[0]
        
        self.backward(self.layers[len(self.layers)-1].backwardsfirst(o, y))
        
        return 1
    def predict(X):
        # return prediction for X
        return 1
    def loss_epochs():
        # evolution of the training error and the validation error for each of the epochs of the system (n_epochs,2)
        return 1    




# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers,0,0,0)

print(nn.forward(np.array([[1], [2], [3], [4]])))    [[1,2,3],[1,2,3]]