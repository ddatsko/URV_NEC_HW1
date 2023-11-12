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
  # Todo: chnage name of activation function to "fact"
  def __init__(self, layers, number_of_epochs, learning_rate, momentum, activationFunctionName, percentage_of_validation):
    self.activationFunctionName = activationFunctionName
    self.L = len(layers)    # number of Layers
    self.n = layers.copy()  # number of neurons in each layer
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.percentage_of_validation = percentage_of_validation

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

    self.delta = [] # values for thresholds
    # for i = 0 there are no thresholds.
    self.delta.append(np.zeros((1, 1)))
    for lay in range(1,self.L):
      self.delta.append(np.zeros(layers[lay]))

    self.d_theta = [] # values for changes to the thresholds
    # for i = 0 there are no thresholds, so no changes for them
    self.d_theta.append(np.zeros((1, 1)))
    for lay in range(1,self.L):
      self.d_theta.append(np.zeros(layers[lay]))
    
    self.d_w = []             # change of edge weights
    # for i = 0 there are no weights.
    self.d_w.append(np.zeros((1, 1)))
    # array is L x numberof neurons in L x number of neurons of L-1
    for lay in range(1, self.L):
      self.d_w.append(np.random.rand(layers[lay], layers[lay - 1]))
      #Todo: this has to be changed

      self.d_w_prev = []   # previois change of edge weights
    # for i = 0 there are no weights, no change of weights so no previios change of weights
    self.d_w_prev.append(np.zeros((1, 1)))
    # array is L x numberof neurons in L x number of neurons of L-1
    for lay in range(1, self.L):
        self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

      
  def fit(self,X,Y):
    # fit forward
    # forumla 1 of BP document
    self.xi[0] = X[0]
    y = Y[0]
    for lay in range(1, self.L):
      for neuron in range(0,self.n[lay]):
        #formula 8
        htemp = 0
        for neuronj in range(0,self.n[lay-1]):
          htemp = htemp + (self.w[lay][neuron][neuronj] * self.xi[(lay-1)][neuronj] )
        self.h[lay][neuron] = htemp - self.theta[lay][neuron]
        #formula 7
        self.xi[lay][neuron] = MyNeuralNetwork.activation_functions[self.activationFunctionName]["forward"](self.h[lay][neuron])

    #calculating deltas for last layer, BP document formula 11
    indexlastLayer = self.L-1 
    for neuroni in range(0,self.n[indexlastLayer]):
      # what happens when z is negative? do I have to take the absolut values when doing  (o(x) - z), I think this not matters we have values in (0,1)
      self.delta[indexlastLayer][neuroni] = MyNeuralNetwork.activation_functions[self.activationFunctionName]['derivative'](self.h[indexlastLayer][neuroni]) * (self.xi[indexlastLayer][neuroni] - y[neuroni])

    # formula 12 
    # we only iterate through layers 1,2,... indextlastyear-1
    for lay in range((indexlastLayer-1),0,-1):
      for neuronj in range(0,self.n[lay]):
        deltatemp = 0;
        for neuroni in range(0,self.n[lay+1]):
          deltatemp = deltatemp + self.delta[lay+1][neuroni] * self.w[lay+1][neuroni][neuronj]
        self.delta[lay][neuronj] = MyNeuralNetwork.activation_functions[self.activationFunctionName]['derivative'](self.h[lay][neuronj]) * deltatemp
      
    # formula 14
    # Todo: build perv in formula momentum * d_w(prev)
    for lay in range(1, self.L):
      for neuroni in range(0,self.n[lay]):
        for neuronj in range(0,self.n[lay-1]):
          self.d_w[lay][neuroni][neuronj] = ((-1)*self.learning_rate) * self.delta[lay][neuroni] * self.xi[lay-1][neuronj] + self.momentum *self.d_w_prev[lay][neuroni][neuronj]

    
     

    return 1
  def feedForward(sample):
    
    return 1
    

# layers include input layer + hidden layers + output layer

layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers,4.0,0.2,0.0,'linear',0.3)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1][0][1], end="\n")

nn.fit([[1,2,3,4]],[[5]])
print("xi = ", nn.xi, end="\n")

print("h = ", nn.h  ,end="\n")
print("delta = ", nn.delta  ,end="\n")

print("d_w = ", nn.d_w  ,end="\n")


