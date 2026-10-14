import numpy as np


class MultilayerPerceptron:
    """Multilayer perceptron with forward propagation using NumPy.

    Architecture: 3 (input) -> 5 -> 5 -> 2 -> 2 (output).
    Layers 1-3 use the sigmoid activation, layer 4 uses ReLU.
    """

    def __init__(self):
        self.net = {}
        self.init_network()

    def init_network(self):
        net = {}
        # layer 1: 3 inputs -> 5 neurons
        net['w1'] = np.array([[0.7, 0.9, 0.3, 0.5, 0.1],
                              [0.2, 0.4, 0.6, 0.8, 0.3],
                              [0.5, 0.1, 0.8, 0.2, 0.6]])
        net['b1'] = np.array([0.5, 0.2, 0.3, 0.1, 0.4])
        # layer 2: 5 -> 5 neurons
        net['w2'] = np.array([[0.2, 0.4, 0.6, 0.1, 0.3],
                              [0.5, 0.3, 0.1, 0.7, 0.2],
                              [0.8, 0.2, 0.4, 0.3, 0.6],
                              [0.1, 0.6, 0.5, 0.2, 0.4],
                              [0.3, 0.7, 0.2, 0.5, 0.1]])
        net['b2'] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # layer 3: 5 -> 2 neurons
        net['w3'] = np.array([[0.1, 0.4],
                              [0.6, 0.2],
                              [0.3, 0.5],
                              [0.7, 0.1],
                              [0.2, 0.8]])
        net['b3'] = np.array([0.2, 0.3])
        # layer 4: 2 -> 2 neurons (output)
        net['w4'] = np.array([[0.5, -0.3],
                              [0.4, 0.6]])
        net['b4'] = np.array([0.1, -0.2])
        self.net = net

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def relu(self, a):
        return np.maximum(0, a)

    def forward(self, x):
        w1, w2, w3, w4 = self.net['w1'], self.net['w2'], self.net['w3'], self.net['w4']
        b1, b2, b3, b4 = self.net['b1'], self.net['b2'], self.net['b3'], self.net['b4']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        z3 = self.sigmoid(a3)

        a4 = np.dot(z3, w4) + b4
        y = self.relu(a4)

        return y


if __name__ == "__main__":
    print("multilayer_perceptron.py defines the MultilayerPerceptron class.")
    print("Architecture: 3 -> 5 -> 5 -> 2 -> 2 (sigmoid, sigmoid, sigmoid, relu)")
    print("Usage:")
    print("    from multilayer_perceptron import MultilayerPerceptron")
    print("    mlp = MultilayerPerceptron()")
    print("    mlp.forward(np.array([1.0, 0.5, -0.5]))")
    print("Run 'python module4.py' to test the class.")
