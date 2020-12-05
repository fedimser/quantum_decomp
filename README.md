# Tool for decomposing unitary matrix into quantum gates

This is a Python tool which takes as input unitary matrix and returns Q# code implementing it. 

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
```

See [example.ipynb](/example.ipynb) for more examples and instructions how to 
use this tool.

### References

* This tool was inspired by [Microsoft Q# Coding Contest](https://codeforces.com/blog/entry/65579) and was implemented as part of online course "[Applications of Quantum Mechanics](https://courses.edx.org/courses/course-v1:MITx+8.06x+1T2019/course/)" at MIT. 

* See this [paper](res/Fedoriaka2019Decomposition.pdf) for detailed description 
of the algorithm and further references to papers with algorithms.
  
* [Blog post](https://codeforces.com/blog/entry/84655) about the tool. 
