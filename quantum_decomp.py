import numpy as np

from src.decompose_2x2 import unitary2x2_to_gates
from src.decompose_4x4 import decompose_4x4_optimal, decompose_4x4_tp
from src.gate import Gate, GateFC, GateSingle, gates_to_matrix
from src.gate2 import Gate2
from src.optimize import optimize_gates
from src.two_level_unitary import TwoLevelUnitary
from src.utils import PAULI_X, is_unitary, is_special_unitary, is_power_of_two


def two_level_decompose(A):
    """Returns list of two-level unitary matrices, which multiply to A.

    Matrices are listed in application order, i.e. if aswer is [u_1, u_2, u_3],
    it means A = u_3 u_2 u_1.
    """
    def make_eliminating_matrix(a, b):
        """Returns unitary matrix U, s.t. [a, b] U = [c, 0].

        Makes second element equal to zero.
        """
        assert (np.abs(a) > 1e-9 and np.abs(b) > 1e-9)
        theta = np.arctan(np.abs(b / a))
        lmbda = -np.angle(a)
        mu = np.pi + np.angle(b) - np.angle(a) - lmbda
        result = np.array([[np.cos(theta) * np.exp(1j * lmbda),
                            np.sin(theta) * np.exp(1j * mu)],
                           [-np.sin(theta) * np.exp(-1j * mu),
                            np.cos(theta) * np.exp(-1j * lmbda)]])
        assert is_special_unitary(result)
        assert np.allclose(np.angle(result[0, 0] * a + result[1, 0] * b), 0)
        assert (np.abs(result[0, 1] * a + result[1, 1] * b) < 1e-9)
        return result

    assert is_unitary(A)
    n = A.shape[0]
    result = []
    # Make a copy, because we are going to mutate it.
    current_A = np.array(A)

    for i in range(n - 2):
        for j in range(n - 1, i, -1):
            if abs(current_A[i, j]) < 1e-9:
                # Element is already zero, skipping.
                pass
            else:
                if abs(current_A[i, j - 1]) < 1e-9:
                    # Just swap columns.
                    u_2x2 = PAULI_X
                else:
                    u_2x2 = make_eliminating_matrix(
                        current_A[i, j - 1], current_A[i, j])
                u_2x2 = TwoLevelUnitary(u_2x2, n, j - 1, j)
                u_2x2.multiply_right(current_A)
                result.append(u_2x2.inv())

    result.append(TwoLevelUnitary(
        current_A[n - 2:n, n - 2:n], n, n - 2, n - 1))
    return result


def two_level_decompose_gray(A):
    """Retunrs list of two-level matrices, which multiplu to A.

    Guarantees that each matrix acts on single bit.
    """
    N = A.shape[0]
    assert is_power_of_two(N)
    assert A.shape == (N, N), "Matrix must be square."
    assert is_unitary(A)

    # Build permutation matrix.
    perm = [x ^ (x // 2) for x in range(N)]  # Gray code.
    P = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        P[i][perm[i]] = 1

    result = two_level_decompose(P @ A @ P.T)
    for matrix in result:
        matrix.apply_permutation(perm)
    return result


def matrix_to_gates(A):
    """Given unitary matrix A, retuns sequence of gates which implements
    action of this matrix on register of qubits.

    Input: A - 2^n x 2^N unitary matrix.
    Returns: sequence of Gate objects.
    """
    matrices = two_level_decompose_gray(A)
    gates = sum([matrix.to_fc_gates() for matrix in matrices], [])
    gates = optimize_gates(gates)
    return gates


def matrix_to_qsharp(A):
    """Given unitary matrix A, retuns Q# code which implements
    action of this matrix on register of qubits called `qs`.

    Input: A - 2^N x 2^N unitary matrix.
    Returns: string - Q# code.
    """
    header = ("operation ApplyUnitaryMatrix (qs : Qubit[]) : Unit {\n"
              "body (...) {\n")
    footer = "  }\n}\n"
    code = '\n'.join(['    ' + gate.to_qsharp_command()
                      for gate in matrix_to_gates(A)])
    return header + code + '\n' + footer
