import unittest
import numpy as np
import quantum_decomp as qd
from scipy.stats import unitary_group


class QuantumDecompTestCase(unittest.TestCase):

    def check_correct_product(self, A, matrices):
        n = A.shape[0]
        B = np.eye(n)
        for matrix in matrices:
            assert matrix.matrix_size == n
            B = matrix.get_full_matrix() @ B
        assert(np.linalg.norm(B - A) < 1e-9)

    def check_acting_on_same_bit(self, matrices):
        for matrix in matrices:
            qd.check_power_of_two(matrix.index1 ^ matrix.index2)

    def check_decompose(self, matrix):
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
        self.check_decompose([[1, 0], [0, 1]])
        self.check_decompose([[0, 1], [1, 0]])
        self.check_decompose([[0, 1j], [1j, 0]])
        self.check_decompose(np.array([[1, 1], [1, -1]] / np.sqrt(2)))

    def test_decompose_3x3(self):
        w = np.exp((2j / 3) * np.pi)
        A = w * np.array([[1, 1, 1], [1, w, w * w],
                          [1, w * w, w]]) / np.sqrt(3)
        self.check_decompose(A)

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
        A = w * np.array([[1, 1, 1, 0], [1, w, w * w, 0],
                          [1, w * w, w, 0], [0, 0, 0, np.sqrt(3)]]) / np.sqrt(3)
        self.check_decompose_gray(A)

    def test_decompose_gray_random(self):
        for matrix_size in [2, 4, 8, 16]:
            for i in range(4):
                A = np.array(unitary_group.rvs(matrix_size))
                self.check_correct_product(A, qd.two_level_decompose(A))

    def test_unitary_to_gates(self):
        for i in range(100):
            A = np.array(unitary_group.rvs(2))
            gates = qd.unitary_to_gates(A)
            assert len(gates) <= 4

            B = np.eye(2)
            for gate in gates:
                B = gate.to_matrix() @ B
            assert np.linalg.norm(A - B) < 1e-9

    def test_TwoLevelUnitary_to_fc_gates(self):
        matrix = qd.TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
        gates = matrix.to_fc_gates()
        assert np.allclose(qd.gates_to_matrix(gates), matrix.get_full_matrix())

    def test_Gate2_to_matrix(self):
        assert np.allclose(qd.Gate2('X').to_matrix(), [[0, 1], [1, 0]])

        assert np.allclose(qd.Gate2('R1', 0).to_matrix(), np.diag([1, 1]))
        assert np.allclose(
            qd.Gate2('R1', np.pi / 2).to_matrix(), np.diag([1, 1j]))
        assert np.allclose(qd.Gate2('R1', np.pi).to_matrix(), np.diag([1, -1]))

        assert np.allclose(qd.Gate2('Ry', 0).to_matrix(), np.diag([1, 1]))
        assert np.allclose(qd.Gate2('Ry', np.pi / 2).to_matrix(),
                           np.array([[1, 1], [-1, 1]]) / np.sqrt(2))
        assert np.allclose(qd.Gate2('Ry', np.pi).to_matrix(),
                           np.array([[0, 1], [-1, 0]]))

        assert np.allclose(qd.Gate2('Rz', 0).to_matrix(), np.diag([1, 1]))
        assert np.allclose(
            qd.Gate2('Rz', np.pi).to_matrix(), np.diag([1j, -1j]))

    def test_GateSingle_to_matrix(self):
        assert np.allclose(qd.GateSingle(qd.Gate2('X'), 0, 2).to_matrix(),
                           [[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])

        assert np.allclose(qd.GateSingle(qd.Gate2('X'), 1, 2).to_matrix(),
                           [[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0]])

    def test_GateFC_to_matrix(self):
        assert np.allclose(qd.GateFC(qd.Gate2('X'), 0, 2).to_matrix(),
                           [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])

        assert np.allclose(qd.GateFC(qd.Gate2('X'), 1, 2).to_matrix(),
                           [[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]])
                            
    def test_matrix_to_gates(self):
        for matrix_size in [2, 4, 8, 16]:
            for i in range(10):
                A = np.array(unitary_group.rvs(matrix_size))
                gates = qd.matrix_to_gates(A)
                assert np.allclose(A, qd.gates_to_matrix(gates))

if __name__ == '__main__':
    unittest.main()
