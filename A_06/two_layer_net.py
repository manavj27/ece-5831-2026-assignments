import numpy as np


class TwoLayerNet:
    """Two-layer neural network trained with numerical gradients.

    Architecture: input -> hidden (sigmoid) -> output (softmax).
    The loss is the cross-entropy error, and the gradient of the loss with
    respect to every weight and bias is found by central differences.
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # avoid overflow
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        # one-hot labels -> class indices
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def _numerical_gradient(self, f, x):
        """Central-difference gradient of f() with respect to array x."""
        h = 1e-4
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]

            x[idx] = tmp_val + h
            fxh1 = f()  # f(x + h)

            x[idx] = tmp_val - h
            fxh2 = f()  # f(x - h)

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val  # restore the original value
            it.iternext()

        return grad

    def numerical_gradient(self, x, t):
        loss_W = lambda: self.loss(x, t)

        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = self._numerical_gradient(loss_W, self.params[key])
        return grads


if __name__ == '__main__':
    print('Usage:')
    print('  from two_layer_net import TwoLayerNet')
    print('  net = TwoLayerNet(input_size=4, hidden_size=10, output_size=3)')
    print('  grads = net.numerical_gradient(x_batch, t_batch)')
    print('  for key in net.params: net.params[key] -= lr * grads[key]')
