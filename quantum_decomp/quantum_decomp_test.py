import warnings

import numpy as np
from scipy.stats import unitary_group, ortho_group

import quantum_decomp as qd
from quantum_decomp.src.gate import gates_to_matrix
from quantum_decomp.src.test_utils import SWAP, check_decomp, QFT_2, CNOT, \
    assert_all_close, random_orthogonal_matrix, random_unitary
from quantum_decomp.src.two_level_unitary import TwoLevelUnitary
from quantum_decomp.src.utils import is_power_of_two


def _check_correct_product(A, matrices):
    n = A.shape[0]
    B = np.eye(n)
    for matrix in matrices:
        assert matrix.matrix_size == n
        B = matrix.get_full_matrix() @ B
    assert np.allclose(A, B)


def _check_acting_on_same_bit(matrices):
    for matrix in matrices:
        assert is_power_of_two(matrix.index1 ^ matrix.index2)


def _check_two_level_decompose(matrix):
    matrix = np.array(matrix)
    _check_correct_product(matrix, qd.two_level_decompose(matrix))


def _check_decompose_gray(matrix):
    matrix = np.array(matrix)
    result = qd.two_level_decompose_gray(matrix)
    _check_correct_product(matrix, result)
    _check_acting_on_same_bit(result)


def _check_matrix_to_gates(mx):
    check_decomp(mx, qd.matrix_to_gates(mx))
    if mx.shape[0] == 4:
        check_decomp(mx, qd.matrix_to_gates(mx, optimize=True))


def test_decompose_2x2():
    _check_two_level_decompose([[1, 0], [0, 1]])
    _check_two_level_decompose([[0, 1], [1, 0]])
    _check_two_level_decompose([[0, 1j], [1j, 0]])
    _check_two_level_decompose(np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def test_decompose_3x3():
    w = np.exp((2j / 3) * np.pi)
    A = w * np.array([[1, 1, 1],
                      [1, w, w * w],
                      [1, w * w, w]]) / np.sqrt(3)
    _check_two_level_decompose(A)


# This test checks that two-level decomposition algorithm ensures that
# diagonal element is equal to 1 after we are done with a row.
def test_diagonal_elements_handled_correctly():
    _check_matrix_to_gates(np.array([
        [1j, 0, 0, 0],
        [0, -1j, 0, 0],
        [0, 0, -1j, 0],
        [0, 0, 0, 1j],
    ]))
    _check_matrix_to_gates(np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1j],
        [0, 0, 1, 0],
        [0, 1j, 0, 0],
    ]))
    _check_matrix_to_gates(np.array([
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [1j, 0, 0, 0],
        [0, 0, 0, 1],
    ]))


def test_decompose_random():
    for matrix_size in range(2, 20):
        for i in range(4):
            A = np.array(unitary_group.rvs(matrix_size))
            _check_correct_product(A, qd.two_level_decompose(A))


def test_decompose_gray_2x2():
    _check_decompose_gray([[1, 0], [0, 1]])
    _check_decompose_gray([[0, 1], [1, 0]])
    _check_decompose_gray([[0, 1j], [1j, 0]])
    _check_decompose_gray(np.array([[1, 1], [1, -1]] / np.sqrt(2)))


def test_decompose_gray_4x4():
    _check_decompose_gray(np.eye(4).T)

    w = np.exp((2j / 3) * np.pi)
    A = w * np.array([[1, 1, 1, 0],
                      [1, w, w * w, 0],
                      [1, w * w, w, 0],
                      [0, 0, 0, np.sqrt(3)]]) / np.sqrt(3)
    _check_decompose_gray(A)


def test_decompose_gray_random():
    for matrix_size in [2, 4, 8, 16]:
        for i in range(4):
            A = np.array(unitary_group.rvs(matrix_size))
            _check_correct_product(A, qd.two_level_decompose(A))


def test_TwoLevelUnitary_inv():
    matrix1 = TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
    matrix2 = matrix1.inv()
    product = matrix1.get_full_matrix() @ matrix2.get_full_matrix()
    assert np.allclose(product, np.eye(8))


def test_TwoLevelUnitary_multiply_right():
    matrix_2x2 = TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
    A1 = unitary_group.rvs(8)
    A2 = np.array(A1)
    matrix_2x2.multiply_right(A1)
    assert np.allclose(A1, A2 @ matrix_2x2.get_full_matrix())


def test_matrix_to_gates_SWAP():
    _check_matrix_to_gates(SWAP)


def test_matrix_to_gates_random_unitary():
    np.random.seed(100)
    for matrix_size in [2, 4, 8, 16]:
        for _ in range(10):
            _check_matrix_to_gates(unitary_group.rvs(matrix_size))


def test_matrix_to_gates_random_orthogonal():
    np.random.seed(100)
    for matrix_size in [2, 4, 8]:
        for _ in range(10):
            _check_matrix_to_gates((ortho_group.rvs(matrix_size)))


def test_matrix_to_gates_identity():
    A = np.eye(16)
    gates = qd.matrix_to_gates(A)
    assert len(gates) == 0


def test_matrix_to_qsharp_SWAP():
    qsharp_code = qd.matrix_to_qsharp(SWAP)
    expected = "\n".join([
        "operation ApplyUnitaryMatrix (qs : Qubit[]) : Unit is Adj {",
        "  CNOT(qs[1], qs[0]);",
        "  CNOT(qs[0], qs[1]);",
        "  CNOT(qs[1], qs[0]);",
        "}", ""])
    assert qsharp_code == expected


def test_matrix_to_cirq_circuit():
    def _check(A):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_all_close(A, qd.matrix_to_cirq_circuit(A).unitary())

    _check(SWAP)
    _check(CNOT)
    _check(QFT_2)

    np.random.seed(100)
    for matrix_size in [2, 4]:
        for _ in range(10):
            _check(random_orthogonal_matrix(matrix_size))
            _check(random_unitary(matrix_size))
    for matrix_size in [8, 16]:
        _check(random_orthogonal_matrix(matrix_size))
        _check(random_unitary(matrix_size))


def test_matrix_to_qiskit_circuit():
    import qiskit.quantum_info as qi

    def _check(matrix):
        circuit = qd.matrix_to_qiskit_circuit(matrix)
        op = qi.Operator(circuit)
        assert np.allclose(op.data, matrix)

    _check(SWAP)
    _check(CNOT)
    _check(QFT_2)

    np.random.seed(100)
    for matrix_size in [2, 4]:
        for _ in range(10):
            _check(random_orthogonal_matrix(matrix_size))
            _check(random_unitary(matrix_size))
    for matrix_size in [8, 16]:
        _check(random_orthogonal_matrix(matrix_size))
        _check(random_unitary(matrix_size))
