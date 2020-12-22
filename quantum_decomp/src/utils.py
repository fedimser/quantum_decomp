import math
import numpy as np

IDENTITY_2x2 = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def is_orthogonal(Q):
    n = Q.shape[0]
    assert Q.shape == (n, n)
    return is_real(Q) and np.allclose(Q.T @ Q, np.eye(n))


def is_unitary(A):
    n = A.shape[0]
    if (A.shape != (n, n)):
        raise ValueError("Matrix is not square.")
    A = np.array(A)
    return np.allclose(np.eye(n), A @ A.conj().T)


def is_real(x):
    return np.allclose(np.real(x), x)


def is_special_unitary(A):
    return is_unitary(A) and np.allclose(np.linalg.det(A), 1.0)


def is_power_of_two(x):
    return (x & (x - 1)) == 0 and x != 0


def cast_to_real(x):
    """Converts complex np.array (known to be real) to real dtype."""
    ans = np.real(x)
    assert np.allclose(x, ans)
    return ans


def skip_identities(gates):
    return [gate for gate in gates if not gate.gate2.is_identity()]


def permute_matrix(A, perm):
    """Applies permutation perm to columns and rows of matrix A.

    Equivalent to:
    P = np.zeros_like(A)
    for i in range(len(perm)): P[i][perm[i]] = 1
    return P @ A @ P.T
    """
    A = np.array(A)
    A[:, :] = A[:, perm]
    A[:, :] = A[perm, :]
    return A
