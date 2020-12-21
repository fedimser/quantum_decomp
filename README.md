# Tool for decomposing unitary matrix into quantum gates

[![PyPI version](https://badge.fury.io/py/quantum-decomp.svg)](https://badge.fury.io/py/quantum-decomp)

This is a Python tool which takes a unitary matrix and returns
a quantum circuit implementing it as Q# code, Cirq circuit, or Qiskit circuit.

### Installing

```
pip install quantum-decomp
```

### Example

```python
>>> import numpy, quantum_decomp
>>> SWAP = numpy.array([[1,0,0,0],[0,0,1,0],[0,1,0,0], [0,0,0,1]])
>>> print(quantum_decomp.matrix_to_qsharp(SWAP, op_name='Swap'))
operation Swap (qs : Qubit[]) : Unit {
  CNOT(qs[1], qs[0]);
  CNOT(qs[0], qs[1]);
  CNOT(qs[1], qs[0]);
}

>>> print(quantum_decomp.matrix_to_cirq_circuit(SWAP))
0: ───@───X───@───
      │   │   │
1: ───X───@───X───

>>> print(quantum_decomp.matrix_to_qiskit_circuit(SWAP))
     ┌───┐     ┌───┐
q_0: ┤ X ├──■──┤ X ├
     └─┬─┘┌─┴─┐└─┬─┘
q_1: ──■──┤ X ├──■──
          └───┘
```

See [example.ipynb](/example.ipynb) for more examples and instructions how to 
use this tool.

### References

* This tool was inspired by [Microsoft Q# Coding Contest](https://codeforces.com/blog/entry/65579) and was implemented as part of online course "[Applications of Quantum Mechanics](https://courses.edx.org/courses/course-v1:MITx+8.06x+1T2019/course/)" at MIT. 

* See this [paper](res/Fedoriaka2019Decomposition.pdf) for detailed description 
of the algorithm and further references to papers with algorithms.
  
* [Blog post](https://codeforces.com/blog/entry/84655) about the tool. 
