import numpy as np

from src.gate2 import Gate2
from src.utils import PAULI_X, check_unitary, check_special_unitary

def su_to_gates(A):
    """Decomposes 2x2 special unitary to gates Ry, Rz.

    R_k(x) = exp(0.5*i*x*sigma_k).
    """
    check_special_unitary(A)
    u00 = A[0, 0]
    u01 = A[0, 1]
    theta = np.arccos(np.abs(u00))
    lmbda = np.angle(u00)
    mu = np.angle(u01)

    result = []
    result.append(Gate2('Rz', lmbda - mu))
    result.append(Gate2('Ry', 2 * theta))
    result.append(Gate2('Rz', lmbda + mu))
    return result


def unitary2x2_to_gates(A):
    """Decomposes 2x2 unitary to gates Ry, Rz, R1.

    R1(x) = diag(1, exp(i*x)).
    """
    check_unitary(A)
    phi = np.angle(np.linalg.det(A))
    if np.abs(phi) < 1e-9:
        return su_to_gates(A)
    elif np.allclose(A, PAULI_X):
        return [Gate2('X')]
    else:
        A = np.diag([1.0, np.exp(-1j * phi)]) @ A
        return su_to_gates(A) + [Gate2('R1', phi)]