import numpy as np
from sklearn.model_selection import train_test_split

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
      self.theta.append(np.random.rand(layers[lay]))

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

    self.d_theta_prev = [] # values for previous changes to the thresholds
    # for i = 0 there are no thresholds, so no changes for them, so no previous changes
    self.d_theta_prev.append(np.zeros((1, 1)))
    for lay in range(1,self.L):
      self.d_theta_prev.append(np.zeros(layers[lay]))
    
    self.d_w = []             # change of edge weights
    # for i = 0 there are no weights.
    self.d_w.append(np.zeros((1, 1)))
    # array is L x numberof neurons in L x number of neurons of L-1
    for lay in range(1, self.L):
      self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
      

      self.d_w_prev = []   # previois change of edge weights
    # for i = 0 there are no weights, no change of weights so no previios change of weights
    self.d_w_prev.append(np.zeros((1, 1)))
    # array is L x numberof neurons in L x number of neurons of L-1
    for lay in range(1, self.L):
        self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

      
  def fit(self,X,Y):
    # split the data
    x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=self.percentage_of_validation,random_state=42)
    self.epoch_error = np.zeros((self.number_of_epochs,2))
    for epoch in range(0,self.number_of_epochs):
      random_indices = np.random.permutation(len(x_train))
      for sample in random_indices:
      
        # print("Now I am doing a iteration with:")
        # print(f"x[{x_train[sample]}]")
        
         self.feedForward(x_train[sample])
         self.backpropagation(y_train[sample])
        # print("xi = ", self.xi, end="\n")

        # print("h = ", self.h  ,end="\n")
        # print("delta = ", self.delta  ,end="\n")

        # print("d_w = ", self.d_w  ,end="\n")
        # print("w = ", self.w  ,end="\n")
        # print("threshold = ", self.theta  ,end="\n")
      self.epoch_error[epoch][0] = self.meanSquaredError(x_train,y_train)
      self.epoch_error[epoch][1]  = self.meanSquaredError(x_val,y_val)
  def meanSquaredError(self,x_data, y_data):
    error_temp = 0
    for i in range(0,len(x_data)):
      self.feedForward(x_data[i])
      for ouput in range(0,self.n[self.L-1]):
        error_temp = error_temp + pow(self.xi[self.L-1][ouput] - y_data[i][ouput],2)
    return (error_temp*0.5)      
       
  def feedForward(self,sample):
    # forumla 1 of BP document
    self.xi[0] = sample
    for lay in range(1, self.L):
      for neuron in range(0,self.n[lay]):
        #formula 8
        htemp = 0
        for neuronj in range(0,self.n[lay-1]):
          htemp = htemp + (self.w[lay][neuron][neuronj] * self.xi[(lay-1)][neuronj] )
        self.h[lay][neuron] = htemp - self.theta[lay][neuron]
        #formula 7
        self.xi[lay][neuron] = MyNeuralNetwork.activation_functions[self.activationFunctionName]["forward"](self.h[lay][neuron])
  def backpropagation(self,y):
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

    for lay in range(1, self.L):
      for neuroni in range(0,self.n[lay]):
        self.d_theta[lay][neuroni] = self.learning_rate * self.delta[lay][neuroni] + self.momentum * self.d_theta_prev[lay][neuroni]

    # formula 15, we update all the weights and thresholds:
    for lay in range(1, self.L):
      for neuroni in range(0,self.n[lay]):
        for neuronj in range(0,self.n[lay-1]):
          self.w[lay][neuroni][neuronj] = self.w[lay][neuroni][neuronj] + self.d_w[lay][neuroni][neuronj]
    
    for lay in range(1, self.L):
      for neuroni in range(0,self.n[lay]):
        self.theta[lay][neuroni] = self.theta[lay][neuroni] + self.d_theta[lay][neuroni]

    self.d_w_prev = [np.copy(arr) for arr in nn.d_w]
    self.d_theta_prev = [np.copy(arr) for arr in nn.d_theta]
  def loss_epochs(self):
    return self.epoch_error
  def predict(self,X):
    prediction = np.zeros((len(X),self.n[self.L-1]))
    for sample in range(0,len(X)):
      self.feedForward(X[sample])
      prediction[sample] = self.xi[self.L-1]
    return prediction


     
  
    

# layers include input layer + hidden layers + output layer

layers = [4,1]
nn = MyNeuralNetwork(layers, 100 ,0.01,0.01,'linear',0.1)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1][0][1], end="\n")
print("threshold = ", nn.theta, end="\n")



#nn.fit([[1,2,3,4],[5,6,7,8],[9,10,11,12],[14,15,16,17]],[[1],[2],[3],[4]])
nn.fit([[1,2,3,4],[5,6,7,8]],[[1],[2]])

errrr = nn.loss_epochs()
print("errror",errrr)

pre= nn.predict([[1,2,3,4],[5,6,7,8]])
print(pre)

s = np.array([1,2,3,4])







