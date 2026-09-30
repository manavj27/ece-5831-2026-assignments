# A_02 – Python NumPy Tutorial

ECE 5831 (2026), Assignment 02.

## Files

| File | Description |
|------|-------------|
| `numpy-tutorials.ipynb` | Jupyter notebook (completed in VS Code) working through the NumPy section of the [CS231n Python NumPy Tutorial](https://cs231n.github.io/python-numpy-tutorial/). All cells have been executed and outputs are saved. |
| `README.md` | This file. |

## Work completed

The notebook covers each required topic, with runnable code and short explanations:

1. **NumPy** – importing the library and checking the version.
2. **Arrays** – creating arrays from lists; `shape`; `zeros`, `ones`, `full`, `eye`, `random.random`.
3. **Array indexing** – slicing (and how slices are views), mixing integer and slice indexing, integer array indexing, and boolean array indexing.
4. **Data types** – inferred vs. explicit `dtype`.
5. **Array math** – elementwise operations (`+ - * /`, `np.sqrt`), `dot` / `@` for vector and matrix products, `np.sum` along axes, and transpose (`.T`).
6. **Broadcasting** – explicit loop vs. `np.tile` vs. broadcasting, the broadcasting rules, and examples (outer product, adding vectors to rows/columns, scalar multiplication).

## How to run

Open `numpy-tutorials.ipynb` in VS Code with the Jupyter extension, select a Python 3 kernel with NumPy installed (`pip install numpy`), and choose **Run All**.
