from logic_gate import LogicGate

INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]

EXPECTED = {
    "AND":  [0, 0, 0, 1],
    "NAND": [1, 1, 1, 0],
    "OR":   [0, 1, 1, 1],
    "NOR":  [1, 0, 0, 0],
    "XOR":  [0, 1, 1, 0],
}


def test_gate(name, gate_fn):
    print(f"{name} gate")
    print("x1 x2 | y  expected")
    passed = True
    for (x1, x2), expected in zip(INPUTS, EXPECTED[name]):
        y = gate_fn(x1, x2)
        mark = "OK" if y == expected else "FAIL"
        print(f" {x1}  {x2} | {y}  {expected}  {mark}")
        passed = passed and y == expected
    print("PASSED\n" if passed else "FAILED\n")
    return passed


def main():
    gate = LogicGate()
    gates = {
        "AND": gate.and_gate,
        "NAND": gate.nand_gate,
        "OR": gate.or_gate,
        "NOR": gate.nor_gate,
        "XOR": gate.xor_gate,
    }
    results = [test_gate(name, fn) for name, fn in gates.items()]
    print(f"{sum(results)}/{len(results)} gates passed")


if __name__ == "__main__":
    main()
