"""
This test verifies that Q# code generated by this library actually compiles to
an operation, which applies specified matrix to a register of qubits.

Also this file demonstrates how you can use this library to generate Q# code
for certain unitary operation, combine it with other Q# code, compile and
execute it - and all this from Python.

To run this test you need to have IQ# and qsharp installed.
One way to achieve that is follow instructions at
https://aka.ms/qdk-install/python and run this test from created conda
environment. You will also need to install libraries imported by this test
(numpy, scipy).

Because this setup is complicated, it's not executed as part of continuous
integration.

Here we just test just few small random matrices. More thorough tests are in
quantum_decomp_test.py.
"""

import re

import os
import numpy as np
import qsharp
from scipy.stats import unitary_group, ortho_group

import quantum_decomp as qd
from quantum_decomp.src.test_utils import QFT_2, SWAP, CNOT

DUMP_CODE = """
open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Extensions.Convert;
open Microsoft.Quantum.Extensions.Math;
open Microsoft.Quantum.Diagnostics;
operation ApplyOp(bits : Bool[], dump_file : String) : Unit {
  let N = Length(bits);
  using (qs = Qubit[N]) {
    ApplyPauliFromBitString(PauliX, true, bits, qs);
    Op(qs);
    DumpMachine(dump_file);
    ResetAll(qs);
  }
}
"""
DUMP_FILE = 'dump_machine.txt'


def read_machine_state():
    """Reads machine state dumped from Q# program (as state vector)."""
    with open(DUMP_FILE) as f:
        lines = f.read().split('\n')[1:-1]
    result = []
    for line in lines:
        expr = re.search(':(.*)==', line)
        if expr:
            expr = expr.group(1)
            expr = expr.replace(' ', '').replace('\t', '')
            expr = expr.replace('i', 'j')
            expr = expr.replace('+-', '-')
            result.append(complex(expr))
    return np.array(result)


def dump_qsharp_unitary(op_code, qubits_count):
    """Returns unitary matrix which is implemented by Q# operation.

    It applies given operation on every possible state vector (using Q#
    simulator) and retrieves state in which operation moves those vectors.
    These states are columns of unitary matrix implemented by the operation.

    args:
        op_code - Q# code for operation, which must be called "Op".
        qubits_count - number of qubits on which operation acts.
    return:
        np.array - unitary matrix implemented by given operation.
    """
    dump_op = qsharp.compile(DUMP_CODE + op_code)[0]
    result = []
    for basis_vector in range(2 ** qubits_count):
        bits = [(basis_vector >> i) % 2 == 1 for i in range(qubits_count)]
        dump_op.simulate(bits=bits, dump_file=DUMP_FILE)
        result.append(read_machine_state())
    os.remove(DUMP_FILE)
    return np.array(result).T


def check_on_matrix(matrix):
    op_code = qd.matrix_to_qsharp(matrix, op_name='Op')
    qubits_count = int(np.log2(matrix.shape[0]))
    dump_matrix = dump_qsharp_unitary(op_code, qubits_count)
    assert np.allclose(matrix, dump_matrix, atol=1e-6)


def test_qsharp_integration_2x2():
    check_on_matrix(unitary_group.rvs(2))
    check_on_matrix(unitary_group.rvs(2))


def test_qsharp_integration_4x4():
    check_on_matrix(SWAP)
    check_on_matrix(CNOT)
    check_on_matrix(QFT_2)
    check_on_matrix(unitary_group.rvs(4))


def test_qsharp_integration_8x8():
    check_on_matrix(unitary_group.rvs(4))


if __name__ == '__main__':
    test_qsharp_integration_2x2()
    test_qsharp_integration_4x4()
    test_qsharp_integration_8x8()
    print("OK!")
