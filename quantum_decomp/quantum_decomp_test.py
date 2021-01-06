import unittest
import warnings

import numpy as np
from scipy.stats import unitary_group, ortho_group

import quantum_decomp as qd
from quantum_decomp.src.gate import gates_to_matrix
from quantum_decomp.src.test_utils import SWAP, check_decomp, QFT_2, CNOT, \
    assert_all_close, random_orthogonal_matrix, random_unitary
from quantum_decomp.src.two_level_unitary import TwoLevelUnitary
from quantum_decomp.src.utils import is_power_of_two


class QuantumDecompTestCase(unittest.TestCase):

    def check_correct_product(self, A, matrices):
        n = A.shape[0]
        B = np.eye(n)
        for matrix in matrices:
            assert matrix.matrix_size == n
            B = matrix.get_full_matrix() @ B
        assert np.allclose(A, B)

    def check_acting_on_same_bit(self, matrices):
        for matrix in matrices:
            assert is_power_of_two(matrix.index1 ^ matrix.index2)

    def check_two_level_decompose(self, matrix):
        matrix = np.array(matrix)
        self.check_correct_product(matrix, qd.two_level_decompose(matrix))

    def check_decompose_gray(self, matrix):
        matrix = np.array(matrix)
        result = qd.two_level_decompose_gray(matrix)
        self.check_correct_product(matrix, result)
        self.check_acting_on_same_bit(result)

    def setUp(self):
        pass

    def test_decompose_2x2(self):
        self.check_two_level_decompose([[1, 0], [0, 1]])
        self.check_two_level_decompose([[0, 1], [1, 0]])
        self.check_two_level_decompose([[0, 1j], [1j, 0]])
        self.check_two_level_decompose(
            np.array([[1, 1], [1, -1]]) / np.sqrt(2))

    def test_decompose_3x3(self):
        w = np.exp((2j / 3) * np.pi)
        A = w * np.array([[1, 1, 1],
                          [1, w, w * w],
                          [1, w * w, w]]) / np.sqrt(3)
        self.check_two_level_decompose(A)

    # This test checks that two-level decomposition algorithm ensures that
    # diagonal element is equal to 1 after we are done with a row.
    def test_diagonal_elements_handled_correctly(self):
        self.check_two_level_decompose(np.array([
            [1j, 0, 0, 0],
            [0, -1j, 0, 0],
            [0, 0, -1j, 0],
            [0, 0, 0, 1j],
        ]))
        self.check_two_level_decompose(np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1j],
            [0, 0, 1, 0],
            [0, 1j, 0, 0],
        ]))
        self.check_two_level_decompose(np.array([
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [1j, 0, 0, 0],
            [0, 0, 0, 1],
        ]))

    def test_decompose_random(self):
        for matrix_size in range(2, 20):
            for i in range(4):
                A = np.array(unitary_group.rvs(matrix_size))
                self.check_correct_product(A, qd.two_level_decompose(A))

    def test_decompose_gray_2x2(self):
        self.check_decompose_gray([[1, 0], [0, 1]])
        self.check_decompose_gray([[0, 1], [1, 0]])
        self.check_decompose_gray([[0, 1j], [1j, 0]])
        self.check_decompose_gray(np.array([[1, 1], [1, -1]] / np.sqrt(2)))

    def test_decompose_gray_4x4(self):
        self.check_decompose_gray(np.eye(4).T)

        w = np.exp((2j / 3) * np.pi)
        A = w * np.array([[1, 1, 1, 0],
                          [1, w, w * w, 0],
                          [1, w * w, w, 0],
                          [0, 0, 0, np.sqrt(3)]]) / np.sqrt(3)
        self.check_decompose_gray(A)

    def test_decompose_gray_random(self):
        for matrix_size in [2, 4, 8, 16]:
            for i in range(4):
                A = np.array(unitary_group.rvs(matrix_size))
                self.check_correct_product(A, qd.two_level_decompose(A))

    def test_TwoLevelUnitary_inv(self):
        matrix1 = TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
        matrix2 = matrix1.inv()
        product = matrix1.get_full_matrix() @ matrix2.get_full_matrix()
        assert np.allclose(product, np.eye(8))

    def test_TwoLevelUnitary_multiply_right(self):
        matrix_2x2 = TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
        A1 = unitary_group.rvs(8)
        A2 = np.array(A1)
        matrix_2x2.multiply_right(A1)
        assert np.allclose(A1, A2 @ matrix_2x2.get_full_matrix())

    def test_matrix_to_gates_SWAP(self):
        gates = qd.matrix_to_gates(SWAP)
        assert np.allclose(SWAP, gates_to_matrix(gates, 2))

    def test_matrix_to_gates_random_unitary(self):
        np.random.seed(100)
        for matrix_size in [2, 4, 8, 16]:
            for _ in range(10):
                A = np.array(unitary_group.rvs(matrix_size))
                check_decomp(A, qd.matrix_to_gates(A))

    def test_matrix_to_gates_random_orthogonal(self):
        np.random.seed(100)
        for matrix_size in [2, 4, 8]:
            for _ in range(10):
                A = np.array(ortho_group.rvs(matrix_size))
                check_decomp(A, qd.matrix_to_gates(A))

    def test_matrix_to_gates_identity(self):
        A = np.eye(16)
        gates = qd.matrix_to_gates(A)

        assert len(gates) == 0

    def test_matrix_to_qsharp_SWAP(self):
        qsharp_code = qd.matrix_to_qsharp(SWAP)

        expected = "\n".join([
            "operation ApplyUnitaryMatrix (qs : Qubit[]) : Unit {",
            "  CNOT(qs[1], qs[0]);",
            "  CNOT(qs[0], qs[1]);",
            "  CNOT(qs[1], qs[0]);",
            "}", ""])
        self.assertEqual(qsharp_code, expected)

    def test_matrix_to_cirq_circuit(self):

        def _check(A):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert_all_close(A, qd.matrix_to_cirq_circuit(A).unitary())

        _check(SWAP)
        _check(CNOT)
        _check(QFT_2)

        np.random.seed(100)
        for matrix_size in [2, 4, 8]:
            _check(random_orthogonal_matrix(matrix_size))
            _check(random_unitary(matrix_size))

    def test_matrix_to_qiskit_circuit(self):
        import qiskit.quantum_info as qi

        def _check(matrix):
            circuit = qd.matrix_to_qiskit_circuit(matrix)
            op = qi.Operator(circuit)
            assert np.allclose(op.data, matrix)

        _check(SWAP)
        _check(CNOT)
        _check(QFT_2)

        np.random.seed(100)
        for matrix_size in [2, 4, 8]:
            _check(random_orthogonal_matrix(matrix_size))
            _check(random_unitary(matrix_size))


if __name__ == '__main__':
    unittest.main()
