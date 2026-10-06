# ECE 5831 - Assignment 3: Perceptron Logic Gates

This folder contains my work for Assignment 3, which implements logic gates
with perceptrons using NumPy. The notebook was completed in Jupyter Notebook
inside Visual Studio Code.

## Files

| File | Description |
|------|-------------|
| `logic_gate.py` | The `LogicGate` class with `and_gate`, `nand_gate`, `or_gate`, `nor_gate` and `xor_gate`. Running `python logic_gate.py` prints how to use the class. |
| `module3.py` | Tests every gate in `LogicGate` against its truth table. Run with `python module3.py`. |
| `module3.ipynb` | Jupyter notebook that walks through each gate, run with outputs saved. |
| `README.md` | This file. It describes the contents of the assignment. |

## How It Works

Each basic gate is a single perceptron: `y = 1` if `np.sum(w * x) + b > 0`,
otherwise `y = 0`.

| Gate | w | b |
|------|---|---|
| AND  | [0.5, 0.5]   | -0.7 |
| NAND | [-0.5, -0.5] | 0.7  |
| OR   | [0.5, 0.5]   | -0.2 |
| NOR  | [-0.5, -0.5] | 0.2  |

A single perceptron cannot do XOR because its outputs are not linearly
separable. XOR is built as a two-layer perceptron:
`XOR(x1, x2) = AND(NAND(x1, x2), OR(x1, x2))`.
