# ECE 5831 - Assignment 2: Python NumPy Tutorial

This folder contains my work for Assignment 2, which follows the
[CS231n Python NumPy Tutorial](https://cs231n.github.io/python-numpy-tutorial/).
The notebook was completed in Jupyter Notebook inside Visual Studio Code.

## Files

| File | Description |
|------|-------------|
| `numpy-tutorials.ipynb` | Jupyter notebook with all of the tutorial's NumPy examples, run with their outputs saved. |
| `README.md` | This file. It describes the contents of the assignment. |

## Work Completed

The notebook covers these sections of the tutorial:

- **NumPy**: importing the library and checking its version.
- **Arrays**: making rank 1 and rank 2 arrays, checking their shape, and using
  `np.zeros`, `np.ones`, `np.full`, `np.eye` and `np.random.random`.
- **Array indexing**: slicing (and slices being views of the original data),
  mixing integer indexing with slices, integer array indexing, and boolean
  array indexing.
- **Data types**: letting NumPy pick the `dtype` and setting one explicitly.
- **Array math**: elementwise operations, `dot` for inner products and
  matrix multiplication, `np.sum` along an axis, and transposing with `.T`.
- **Broadcasting**: adding a vector to each row of a matrix three ways
  (explicit loop, `np.tile`, broadcasting), plus outer products, adding a
  vector to each column, and multiplying by a scalar.
