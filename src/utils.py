import math
import numpy as np

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def is_unitary(A):
    n = A.shape[0]
    if (A.shape != (n, n)):
        raise ValueError("Matrix is not square.")
    A = np.array(A)
    return np.allclose(np.eye(n), A @ A.conj().T)


def is_special_unitary(A):
    return is_unitary(A) and np.allclose(np.linalg.det(A), 1.0)


def is_power_of_two(x):
    return 2**int(math.log2(x)) == x


def cast_to_real(x):
    """Converts complex np.array (known to be real) to real dtype."""
    ans = np.real(x)
    assert np.allclose(x, ans)
    return ans
