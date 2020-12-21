import numpy as np
from scipy.stats import unitary_group

from quantum_decomp.src.gate import gates_to_matrix

SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
QFT_2 = 0.5 * np.array([[1, 1, 1, 1],
                        [1, 1j, -1, -1j],
                        [1, -1, 1, -1],
                        [1, -1j, -1, 1j]])


def random_unitary(n):
    return unitary_group.rvs(n)


def random_special_unitary(n):
    A = unitary_group.rvs(n)
    return A * np.linalg.det(A) ** (-1 / n)


def random_orthogonal_matrix(n):
    return unitary_group.rvs(n)


def assert_all_close(x, y, tol=1e-9):
    diff = np.abs(x - y)
    if np.max(diff) > tol:
        raise AssertionError(
            'Not close:\nx=%s\ny=%s\ndiff=%s' %
            (x, y, diff))


def check_decomp(matrix, gates, tol=1e-9):
    """Checks that `gates` is decomposition of `matrix`."""
    qubits_count = int(np.log2(matrix.shape[0]))
    assert_all_close(matrix, gates_to_matrix(gates, qubits_count), tol=tol)
