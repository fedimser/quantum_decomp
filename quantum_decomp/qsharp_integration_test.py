"""
This test verifies that Q# code generated by this library actually compiles to
an operation, which applies specified matrix to a register of qubits.
"""

import io
import tempfile
from contextlib import redirect_stdout

import cirq
import numpy as np
import qsharp
from scipy.stats import unitary_group

import quantum_decomp as qd
from quantum_decomp.src.test_utils import CNOT, QFT_2, SWAP
from quantum_decomp.src.utils import permute_matrix


def change_int_endianness(val, n):
    return sum(((val >> i) & 1) << (n - 1 - i) for i in range(n))


def change_matrix_endianness(matrix, n):
    assert matrix.shape == (2**n, 2**n)
    perm = [change_int_endianness(i, n) for i in range(2**n)]
    return permute_matrix(matrix, perm)


def dump_qsharp_unitary(op_code, qubits_count):
    """Returns unitary matrix which is implemented by Q# operation.

    args:
        op_code - Q# code for operation, which must be called "Op".
        qubits_count - number of qubits on which operation acts.
    return:
        np.array - unitary matrix implemented by given operation.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(tmp_dir + "/qsharp.json", "w") as f:
            f.write('{"files":["Op.qs"]}')
        with open(tmp_dir + "/Op.qs", "w") as f:
            f.write(op_code)
        qsharp.init(project_root=tmp_dir,
                    target_profile=qsharp.TargetProfile.Base)
        f = io.StringIO()
        with redirect_stdout(f):
            qsharp.eval(
                f'Std.Diagnostics.DumpOperation({qubits_count}, Op.Op);')
        dump_output = f.getvalue()

    tokens = dump_output.replace('𝑖', 'j').replace('−', '-').split(' ')[1:]
    values = np.array([np.complex128(x) for x in tokens], dtype=np.complex128)
    assert len(values == (2**qubits_count)**2)
    ans = values.reshape((2**qubits_count, 2**qubits_count))
    return change_matrix_endianness(ans, qubits_count)


def check_on_matrix(matrix):
    op_code = qd.matrix_to_qsharp(matrix, op_name='Op')
    qubits_count = int(np.log2(matrix.shape[0]))
    dump_matrix = dump_qsharp_unitary(op_code, qubits_count)
    assert cirq.equal_up_to_global_phase(matrix, dump_matrix, atol=2e-4)


def test_qsharp_integration_2x2():
    for _ in range(10):
        check_on_matrix(unitary_group.rvs(2))


def test_qsharp_integration_4x4():
    check_on_matrix(SWAP)
    check_on_matrix(CNOT)
    check_on_matrix(QFT_2)
    for _ in range(10):
        check_on_matrix(unitary_group.rvs(4))


def test_qsharp_integration_8x8():
    for _ in range(10):
        check_on_matrix(unitary_group.rvs(8))


def test_qsharp_integration_16x16():
    for _ in range(10):
        check_on_matrix(unitary_group.rvs(16))
