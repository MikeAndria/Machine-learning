import numpy as np
from matplotlib import pyplot as plt


class BinaryClassifier:
    def __init__(self, n_iterations = 1000, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.W = None
        self.biais = None
        self.costs = []

    def cost_function(self, y, y_predicted):
        y = np.array(y)
        y = y.reshape(-1, 1)
        m = y.shape[0]
        epsilone = 1e-10
        cost = -(1 / m) * np.sum(y * np.log(y_predicted + epsilone) + (1 - y) * np.log(1 - y_predicted + epsilone))
        return cost
        
    def sigmoide(self, z):
        '''z[z > 700] = 1
        z[z < 0] = 0'''
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        y = np.array(y)
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))
        self.biais = 0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.W) + self.biais
            y_predicted = self.sigmoide(linear_model)

            #Calcul de cost
            cost = self.cost_function(y, y_predicted)
            self.costs.append(cost)

            #Calcul des gradients
            dW = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #MAJ des parametres
            self.W = self.W - self.learning_rate * dW
            self.biais = self.biais - self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.W) + self.biais
        y_predicted = self.sigmoide(linear_model)
        y_predicted = y_predicted > 0.5
        return np.array(y_predicted, dtype='int')

    def accuracy(self, X, y):
        y = np.array(y)
        y = y.reshape(-1, 1)
        y_predicted = self.predict(X)
        acc = np.mean(y_predicted == y)
        return acc*100       

    def cost_plot(self):
        plt.figure()
        plt.plot(self.costs)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()