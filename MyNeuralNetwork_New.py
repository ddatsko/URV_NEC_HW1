import numpy as np
from sklearn.model_selection import train_test_split


# Neural Network class
class MyNeuralNetwork:
    activation_functions_ = {
        'relu': {"forward": lambda x: np.maximum(x, 0), "derivative": lambda x: (x > 0) * 1},
        'linear': {"forward": lambda x: x, "derivative": lambda x: np.ones_like(x)},
        'sigmoid': {"forward": lambda x: 1 / (1 + np.exp(-x)),
                    "derivative": lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))},
        'tanh': {"forward": lambda x: np.tanh(x), "derivative": lambda x: 1 - np.tanh(x) ** 2}
    }

    # Todo: chnage name of activation function to "fact"
    def __init__(self, layers, number_of_epochs, learning_rate, momentum, activation_function_name,
                 percentage_of_validation):
        self.activation = self.activation_functions_[activation_function_name]["forward"]
        self.activation_derivative = self.activation_functions_[activation_function_name]["derivative"]
        self.L = len(layers)  # number of Layers
        self.n = layers.copy()  # number of neurons in each layer
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.percentage_of_validation = percentage_of_validation

        self.h = []  # array of array of fields
        # for i = 0 there are no fields.
        self.h.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.h.append(np.zeros(layers[lay]))

        self.xi = []  # node values
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        self.w = []  # edge weights
        # for i = 0 there are no weights.
        self.w.append(np.zeros((1, 1)))
        # array is L x numberof neurons in L x number of neurons of L-1
        for lay in range(1, self.L):
            # TODO: make it dependent on activation functions. Here, we use He initialization, which is good for ReLu

            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2 / self.n[lay - 1]))

        self.theta = []  # values for thresholds
        # for i = 0 there are no thresholds.
        self.theta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.theta.append(np.random.randn(layers[lay], 1) * np.sqrt(2 / self.n[lay - 1]))

        self.delta = []  # values for thresholds
        # for i = 0 there are no thresholds.
        self.delta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.delta.append(np.zeros(layers[lay]))

        self.d_theta = []  # values for changes to the thresholds
        # for i = 0 there are no thresholds, so no changes for them
        self.d_theta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_theta.append(np.zeros(layers[lay]))

        self.d_theta_prev = []  # values for previous changes to the thresholds
        # for i = 0 there are no thresholds, so no changes for them, so no previous changes
        self.d_theta_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_theta_prev.append(np.zeros((layers[lay], 1)))

        self.d_w = []  # change of edge weights
        # for i = 0 there are no weights.
        self.d_w.append(np.zeros((1, 1)))
        # array is L x numberof neurons in L x number of neurons of L-1
        for lay in range(1, self.L):
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))

            self.d_w_prev = []  # previois change of edge weights
        # for i = 0 there are no weights, no change of weights so no previios change of weights
        self.d_w_prev.append(np.zeros((1, 1)))
        # array is L x numberof neurons in L x number of neurons of L-1
        for lay in range(1, self.L):
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

    def fit(self, X, Y, batch_size=1):
        # split the data
        # np.random.seed(39)
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=self.percentage_of_validation)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        # Due to weights placement, Xs samples have to be columns

        self.epoch_error = np.zeros((self.number_of_epochs, 2))
        for epoch in range(0, self.number_of_epochs):
            random_indices = np.random.permutation(x_train.shape[0])
            # np.random.shuffle(x_train)
            for i in range(0, len(random_indices), batch_size):
                self.feed_forward(x_train[random_indices[i:i + batch_size], :].T)
                self.backpropagation(y_train[random_indices[i:i + batch_size]].T)

            self.epoch_error[epoch][0] = self.mean_squared_error(x_train, y_train)
            self.epoch_error[epoch][1] = self.mean_squared_error(x_val, y_val)

    def mean_squared_error(self, x_data, y_data):
        prediction = self.predict(x_data)
        return np.mean((prediction.flatten() - y_data.flatten()) ** 2)

    def backpropagation(self, y):
        # calculating deltas for last layer, BP document formula 11
        last_layer_index = self.L - 1
        self.delta[last_layer_index] = (
                self.activation_derivative(self.h[last_layer_index]) * (self.xi[last_layer_index] - y)).T

        # we only iterate through layers 1,2,... indextlastyear-1
        for lay in range(last_layer_index, 0, -1):
            # formula 12
            if lay != last_layer_index:
                self.delta[lay] = self.activation_derivative(self.h[lay]).T * (self.delta[lay + 1] @ self.w[lay + 1])

            # formula 14
            self.d_w[lay] = -self.learning_rate * (self.xi[lay - 1] @ self.delta[lay]).T + self.momentum * \
                            self.d_w_prev[lay]
            self.d_theta[lay] = self.learning_rate * np.sum(self.delta[lay].T, axis=1, keepdims=True) + self.momentum * self.d_theta_prev[lay]

        # formula 15, we update all the weights and thresholds:
        for lay in range(1, self.L):
            self.w[lay] += self.d_w[lay]
            self.theta[lay] += self.d_theta[lay]

        # TODO: check if copies are needed here
        self.d_w_prev = [np.copy(arr) for arr in self.d_w]
        self.d_theta_prev = [np.copy(arr) for arr in self.d_theta]

    def loss_epochs(self):
        return self.epoch_error

    def predict(self, X):
        prediction = X.T
        for lay in range(1, self.L):
            prediction = self.activation(self.w[lay] @ prediction - self.theta[lay])
        return prediction.flatten()

    def feed_forward(self, sample):
        # forumla 1 of BP document
        self.xi[0] = sample
        for lay in range(1, self.L):
            self.h[lay] = self.w[lay] @ self.xi[lay - 1] - self.theta[lay]
            self.xi[lay] = self.activation(self.h[lay])

