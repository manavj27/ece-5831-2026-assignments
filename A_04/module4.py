import numpy as np
from multilayer_perceptron import MultilayerPerceptron

INPUTS = [
    np.array([1.0, 0.5, -0.5]),
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
    np.array([-1.0, 2.0, 0.3]),
]


def main():
    mlp = MultilayerPerceptron()

    print("Network shapes")
    for key, value in mlp.net.items():
        print(f"  {key}: {value.shape}")
    print()

    for x in INPUTS:
        y = mlp.forward(x)
        print(f"x = {x} -> y = {y}")
    print()

    # forward propagation also works on a batch of inputs
    batch = np.array(INPUTS)
    y_batch = mlp.forward(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {y_batch.shape}")
    print(y_batch)


if __name__ == "__main__":
    main()
