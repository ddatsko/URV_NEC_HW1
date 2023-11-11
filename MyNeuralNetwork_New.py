import numpy as np


# Neural Network class
class MyNeuralNetwork:

  activation_functions = {
        'relu': {"forward": lambda x: np.maximum(x, 0), "derivative": lambda x: (x > 0) * 1},
        'linear': {"forward": lambda x: x, "derivative": lambda _: 1},
        'sigmoid': {"forward": lambda x:1/(1+np.exp(-x)), "derivative":lambda x: (1/(1+np.exp(-x))) * (1 - (1/(1+np.exp(-x)))) }
        # TODO: add more activation functions here. We can move the implementations to a separate file.
        #  Also think if this design is good enough, maybe wrap into some classes
    }
  
  def __init__(self, layers, activationFunctionName):
    self.activationFunctionName = activationFunctionName
    self.L = len(layers)    # number of Layers
    self.n = layers.copy()  # number of neurons in each layer

    self.h = []  # array of array of fields
    # for i = 0 there are no fields.
    self.h.append(np.zeros((1, 1)))
    for lay in range(1,self.L):
      self.h.append(np.zeros(layers[lay]))
    
    self.xi = []            # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []             # edge weights
    # for i = 0 there are no weights.
    self.w.append(np.zeros((1, 1)))
    # array is L x numberof neurons in L x number of neurons of L-1
    for lay in range(1, self.L):
      self.w.append(np.random.rand(layers[lay], layers[lay - 1]))
      #Todo: this has to be changed

    self.theta = [] # values for thresholds
    # for i = 0 there are no thresholds.
    self.theta.append(np.zeros((1, 1)))
    for lay in range(1,self.L):
      self.theta.append(np.zeros(layers[lay]))

    self.delta = []
      
  def fit(self,X,y):
    # fit forward
    # forumla 1 of BP document
    self.xi[0] = X[0]
    
    for lay in range(1, self.L):
      for neuron in range(0,self.n[lay]):
        #formula 8
        htemp = 0
        for neuronj in range(0,self.n[lay-1]):
          htemp = htemp + (self.w[lay][neuron][neuronj] * self.xi[(lay-1)][neuronj] )
        self.h[lay][neuron] = htemp - self.theta[lay][neuron]
        #formula 7
        self.xi[lay][neuron] = MyNeuralNetwork.activation_functions[self.activationFunctionName]["forward"](self.h[lay][neuron])
     
    for lay in range(10,0,-1):
     print(lay)
     

    return 1
  def feedForward(sample):
    
    return 1
    

# layers include input layer + hidden layers + output layer

layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers,'linear')

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1][0][1], end="\n")

nn.fit([[1,2,3,4]],[5])
print("xi = ", nn.xi, end="\n")

