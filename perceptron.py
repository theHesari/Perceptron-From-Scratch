import numpy as np

class Perceptron:

    def __init__(self, learning_rate, epochs):

        self.bias = None
        self.weights = None
        self.learning_rate= learning_rate
        self.epochs = epochs

    def activation(self, z):
        return np.heaviside(z, 0)

    def fit(self, X, y):

        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(len(X)):
                z = np.dot(X, self.weights) + self.bias
                y_pred = self.activation(z)

                self.weights = self.weights + self.learning_rate*(y[i] - y_pred[i])*X[i]
                self.bias = self.bias + self.learning_rate*(y[i] - y_pred[i])
        
        return self.weights, self.bias


    
    def predict(self, X):

        z = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(z)

        return y_pred


