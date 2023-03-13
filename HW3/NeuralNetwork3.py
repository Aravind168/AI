import argparse

import numpy as np

LAYERS = [2,16,16,1]
np.random.seed(7)

def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def cost_derivative(a, y):
    return (a-y)

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # for w in self.weights:
        #     print('w : ',w.shape)
        # for b in self.weights:
        #     print('b :', b.shape)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a=a.reshape(-1,1)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0):
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        x=x.reshape(-1,1)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        # print(cost_derivative(activations[-1], y).shape)
        delta = cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            print(delta.shape)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def predict(self, data):
        results = [np.where(self.feedforward(x)>=0.5, 1, 0) for x in data]
        results=np.array(results)
        np.savetxt('test_predictions.csv', results.flatten(), delimiter=',', fmt="%d")

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data')
    parser.add_argument('train_labels')
    parser.add_argument('test_data')
    args = parser.parse_args()

    training_data = np.genfromtxt(args.train_data, delimiter=',')
    training_labels = np.genfromtxt(args.train_labels, delimiter=',', dtype=int).reshape(-1,1)
    testing_data = np.genfromtxt(args.test_data, delimiter=',')
        
    newDS = []
    for i in training_data:
        newDS.append(np.array([i[0], i[1], np.sin(i[0]), np.sin(i[1])]))
    
    train_data = []
    for i in range(len(training_data)):
        train_data.append((training_data[i], training_labels[i]))
        
    model = Network(LAYERS)
    '''test'''
    model.SGD(train_data, 1, 1, 0.1, 0.001)
    model.predict(newDS)
    ''''''
