import numpy as np
from scipy.stats import unitary_group, ortho_group

from quantum_decomp.src.decompose_4x4 import decompose_4x4_optimal, \
    decompose_4x4_tp, decompose_product_state
from quantum_decomp.src.test_utils import SWAP, CNOT, QFT_2, check_decomp, \
    assert_all_close, random_special_unitary


def test_decompose_4x4_optimal_corner_cases():
    check_decomp(SWAP, decompose_4x4_optimal(SWAP))
    check_decomp(CNOT, decompose_4x4_optimal(CNOT))
    check_decomp(QFT_2, decompose_4x4_optimal(QFT_2))

    w = np.exp((2j / 3) * np.pi)
    A = w * np.array([[1, 1, 1, 0],
                      [1, w, w * w, 0],
                      [1, w * w, w, 0],
                      [0, 0, 0, np.sqrt(3)]]) / np.sqrt(3)
    check_decomp(A, decompose_4x4_optimal(A), tol=3e-8)

    Phi = np.sqrt(0.5) * np.array([[1, -1j, 0, 0],
                                   [0, 0, -1j, 1],
                                   [0, 0, -1j, -1],
                                   [1, 1j, 0, 0]])
    check_decomp(Phi, decompose_4x4_optimal(Phi))


def test_decompose_4x4_optimal_tensor_products():
    Id = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    ops = [Id, X, Y, Z, H]

    for m1 in ops:
        for m2 in ops:
            A = np.kron(m1, m2)
            check_decomp(A, decompose_4x4_optimal(A), tol=2e-9)


def test_decompose_4x4_optimal_random_unitary():
    np.random.seed(100)
    for _ in range(10):
        A = unitary_group.rvs(4)
        check_decomp(A, decompose_4x4_optimal(A))


def test_decompose_4x4_optimal_random_orthogonal():
    np.random.seed(100)
    for _ in range(10):
        A = ortho_group.rvs(4)
        check_decomp(A, decompose_4x4_optimal(A))


def test_decompose_4x4_tp():
    np.random.seed(100)
    for _ in range(10):
        U = np.kron(random_special_unitary(2), random_special_unitary(2))
        A, B = decompose_4x4_tp(U)
        assert_all_close(U, np.kron(A, B))


def test_decompose_product_state():
    def _check(state):
        a, b = decompose_product_state(np.array(state))
        assert np.allclose(np.kron(a, b), state)

    _check([0, -3e-49j, -np.sqrt(0.5), -np.sqrt(0.5)])
    _check([-6.29490599e-09 - 7.85046229e-17j,
            0, 0, -1e-09 + 1.00000000e+00j])
