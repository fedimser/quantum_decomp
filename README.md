# Tool for decomposing unitary matrix into quantum gates.

This is a python tool which takes as input unitary matrix and returns Q# code implementing it. 

See [example.ipynb](/example.ipynb) on instructions how to use it.

Code in qsharp/ verifies that output produced by this tool can be plugged into Q# program and produces expected reult. Some Q# code for dumping matrix was taken from [QuantumKatas](https://github.com/Microsoft/QuantumKatas) repository.

This tool was inspired by [Microsoft Q# Coding Contest](https://codeforces.com/blog/entry/65579) and was implemented as part of online course "[Applications of Quantum Mechanics](https://courses.edx.org/courses/course-v1:MITx+8.06x+1T2019/course/)" at MIT. 

See this [paper](res/Fedoriaka2019Decomposition.pdf) for detailed description of the algorithm.
