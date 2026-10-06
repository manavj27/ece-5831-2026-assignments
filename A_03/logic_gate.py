import numpy as np


class LogicGate:
    """Logic gates built from single-layer perceptrons using NumPy.

    Each basic gate computes y = 1 if sum(w * x) + b > 0 else 0.
    XOR is not linearly separable, so it is built from two layers:
    XOR(x1, x2) = AND(NAND(x1, x2), OR(x1, x2)).
    """

    def __init__(self):
        pass

    def _perceptron(self, x1, x2, w, b):
        x = np.array([x1, x2])
        y = np.sum(w * x) + b
        return 1 if y > 0 else 0

    def and_gate(self, x1, x2):
        w = np.array([0.5, 0.5])
        b = -0.7
        return self._perceptron(x1, x2, w, b)

    def nand_gate(self, x1, x2):
        w = np.array([-0.5, -0.5])
        b = 0.7
        return self._perceptron(x1, x2, w, b)

    def or_gate(self, x1, x2):
        w = np.array([0.5, 0.5])
        b = -0.2
        return self._perceptron(x1, x2, w, b)

    def nor_gate(self, x1, x2):
        w = np.array([-0.5, -0.5])
        b = 0.2
        return self._perceptron(x1, x2, w, b)

    def xor_gate(self, x1, x2):
        s1 = self.nand_gate(x1, x2)
        s2 = self.or_gate(x1, x2)
        return self.and_gate(s1, s2)


if __name__ == "__main__":
    print("logic_gate.py defines the LogicGate class.")
    print("Gates: and_gate, nand_gate, or_gate, nor_gate, xor_gate")
    print("Usage:")
    print("    from logic_gate import LogicGate")
    print("    gate = LogicGate()")
    print("    gate.and_gate(1, 1)  # returns 1")
    print("Run 'python module3.py' to test every gate.")
