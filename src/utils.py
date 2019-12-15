import math
import numpy as np

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

def check_unitary(A):
    n = A.shape[0]
    if (A.shape != (n, n)):
        raise ValueError("Matrix is not square.")
    A = np.array(A)
    if (np.linalg.norm(np.eye(n) - A @ A.conj().T) > 1e-9):
        raise ValueError("Matrix is not unitary: %s" % str(A))
        
def check_special_unitary(A):
    check_unitary(A)
    assert np.abs(np.linalg.det(A) - 1.0) < 1e-9

def check_power_of_two(x):
    if 2**int(math.log2(x)) != x:
        raise ValueError(str(x) + " is not a power of two.")