import numpy as np

from quantum_decomp.src.utils import PAULI_X, is_unitary, IDENTITY_2x2


class TwoLevelUnitary:
    """Represents two-level unitary matrix.

    Two-level unitary matrix is a unitary matrix obtained from the identity
    matrix by changing a 2x2 principal submatrix.
    """

    def __init__(self, matrix2x2, matrix_size, index1, index2):
        assert index1 != index2
        assert index1 < matrix_size and index2 < matrix_size
        assert matrix2x2.shape == (2, 2)
        assert is_unitary(matrix2x2)

        self.matrix_size = matrix_size
        self.index1 = index1
        self.index2 = index2
        self.matrix_2x2 = matrix2x2
        self.order_indices()

    def __repr__(self):
        self.order_indices()
        return "%s on (%d, %d)" % (
            str(self.matrix_2x2), self.index1, self.index2)

    def order_indices(self):
        if self.index1 > self.index2:
            self.index1, self.index2 = self.index2, self.index1
            self.matrix_2x2 = PAULI_X @ self.matrix_2x2 @ PAULI_X

    def get_full_matrix(self):
        matrix_full = np.eye(self.matrix_size, dtype=np.complex128)
        matrix_full[self.index1, self.index1] = self.matrix_2x2[0, 0]
        matrix_full[self.index1, self.index2] = self.matrix_2x2[0, 1]
        matrix_full[self.index2, self.index1] = self.matrix_2x2[1, 0]
        matrix_full[self.index2, self.index2] = self.matrix_2x2[1, 1]
        return matrix_full

    def multiply_right(self, A):
        """M.multiply_right(A) is equivalent to A = A @ M.get_full_matrix()."""
        idx = (self.index1, self.index2)
        A[:, idx] = A[:, idx] @ self.matrix_2x2

    def inv(self):
        return TwoLevelUnitary(self.matrix_2x2.conj().T,
                               self.matrix_size,
                               self.index1,
                               self.index2)

    def apply_permutation(self, perm):
        assert(len(perm) == self.matrix_size)
        self.index1 = perm[self.index1]
        self.index2 = perm[self.index2]

    def is_identity(self):
        return np.allclose(self.matrix_2x2, IDENTITY_2x2)
