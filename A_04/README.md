# ECE 5831 - Assignment 4: Multilayer Perceptron

This folder contains my work for Assignment 4, which implements forward
propagation through a multilayer perceptron using NumPy.

## Files

| File | Description |
|------|-------------|
| `multilayer_perceptron.py` | The `MultilayerPerceptron` class with `init_network`, `sigmoid`, `relu` and `forward`. Running `python multilayer_perceptron.py` prints how to use the class. |
| `module4.py` | Imports the class and runs forward propagation on example inputs. Run with `python module4.py`. |
| `module4.ipynb` | Jupyter notebook that shows the forward-propagation results layer by layer, run with outputs saved. |
| `README.md` | This file. It describes the contents of the assignment. |

## Network

| Layer | Neurons | Weight shape | Activation |
|-------|---------|--------------|------------|
| Input | 3 | - | - |
| 1 | 5 | (3, 5) | Sigmoid |
| 2 | 5 | (5, 5) | Sigmoid |
| 3 | 2 | (5, 2) | Sigmoid |
| 4 (output) | 2 | (2, 2) | ReLU |

Each layer computes `a = np.dot(x, W) + b` and then applies its activation.
The weights and biases are fixed values chosen by hand.
