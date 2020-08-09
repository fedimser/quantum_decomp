import unittest
import numpy as np
import warnings
import quantum_decomp as qd

from scipy.stats import unitary_group, ortho_group

from src.decompose_4x4 import decompose_product_state

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


class QuantumDecompTestCase(unittest.TestCase):

    def _random_su(self, n):
        A = unitary_group.rvs(n)
        return A * np.linalg.det(A) ** (-1 / n)

    def assertAllClose(self, x, y, tol=1e-9):
        diff = np.abs(x - y)
        if np.max(diff) > tol:
            raise AssertionError(
                'Not close:\nx=%s\ny=%s\ndiff=%s' %
                (x, y, diff))

    def check_correct_product(self, A, matrices):
        n = A.shape[0]
        B = np.eye(n)
        for matrix in matrices:
            assert matrix.matrix_size == n
            B = matrix.get_full_matrix() @ B
        assert np.allclose(A, B)

    def check_acting_on_same_bit(self, matrices):
        for matrix in matrices:
            assert qd.is_power_of_two(matrix.index1 ^ matrix.index2)

    def check_decomp(self, matrix, gates, tol=1e-9):
        """Checks that `gates` is decomposition of `matrix`."""
        self.assertAllClose(matrix, qd.gates_to_matrix(gates), tol=tol)

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
        A = w * np.array([[1, 1, 1],
                          [1, w, w * w],
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

    def test_unitary2x2_to_gates(self):
        for i in range(100):
            A = np.array(unitary_group.rvs(2))
            gates = qd.unitary2x2_to_gates(A)
            assert len(gates) <= 4

            B = np.eye(2)
            for gate in gates:
                B = gate.to_matrix() @ B
            assert np.linalg.norm(A - B) < 1e-9

    def test_TwoLevelUnitary_to_fc_gates(self):
        matrix = qd.TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
        gates = matrix.to_fc_gates()
        assert np.allclose(qd.gates_to_matrix(gates), matrix.get_full_matrix())

    def test_TwoLevelUnitary_inv(self):
        matrix1 = qd.TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
        matrix2 = matrix1.inv()
        product = matrix1.get_full_matrix() @ matrix2.get_full_matrix()
        assert np.allclose(product, np.eye(8))

    def test_TwoLevelUnitary_multiply_right(self):
        matrix_2x2 = qd.TwoLevelUnitary(unitary_group.rvs(2), 8, 1, 5)
        A1 = unitary_group.rvs(8)
        A2 = np.array(A1)
        matrix_2x2.multiply_right(A1)
        assert np.allclose(A1, A2 @ matrix_2x2.get_full_matrix())

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

    def test_matrix_to_gates_SWAP(self):
        gates = qd.matrix_to_gates(SWAP)
        assert np.allclose(SWAP, qd.gates_to_matrix(gates))

    def test_matrix_to_gates_random_unitary(self):
        np.random.seed(100)
        for matrix_size in [2, 4, 8, 16]:
            for _ in range(10):
                A = np.array(unitary_group.rvs(matrix_size))
                self.check_decomp(A, qd.matrix_to_gates(A))

    def test_matrix_to_gates_random_orthogonal(self):
        np.random.seed(100)
        for matrix_size in [2, 4, 8]:
            for _ in range(10):
                A = np.array(ortho_group.rvs(matrix_size))
                self.check_decomp(A, qd.matrix_to_gates(A))

    def test_matrix_to_gates_identity(self):
        A = np.eye(16)
        gates = qd.matrix_to_gates(A)

        assert len(gates) == 0

    def test_matrix_to_qsharp_SWAP(self):
        qsharp_code = qd.matrix_to_qsharp(SWAP)

        expected = "\n".join([
            "operation ApplyUnitaryMatrix (qs : Qubit[]) : Unit {",
            "body (...) {",
            "    CNOT(qs[1], qs[0]);",
            "    CNOT(qs[0], qs[1]);",
            "    CNOT(qs[1], qs[0]);",
            "  }",
            "}", ""])
        self.assertEqual(qsharp_code, expected)

    def test_decompose_4x4_optimal_corner_cases(self):
        self.check_decomp(SWAP, qd.decompose_4x4_optimal(SWAP))
        self.check_decomp(CNOT, qd.decompose_4x4_optimal(CNOT))
        self.check_decomp(QFT_2, qd.decompose_4x4_optimal(QFT_2))

        w = np.exp((2j / 3) * np.pi)
        A = w * np.array([[1, 1, 1, 0],
                          [1, w, w * w, 0],
                          [1, w * w, w, 0],
                          [0, 0, 0, np.sqrt(3)]]) / np.sqrt(3)
        self.check_decomp(A, qd.decompose_4x4_optimal(A), tol=3e-8)

        Phi = np.sqrt(0.5) * np.array([[1, -1j, 0, 0],
                                       [0, 0, -1j, 1],
                                       [0, 0, -1j, -1],
                                       [1, 1j, 0, 0]])
        self.check_decomp(Phi, qd.decompose_4x4_optimal(Phi))

    def test_decompose_4x4_optimal_tensor_products(self):
        Id = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        ops = [Id, X, Y, Z, H]

        for m1 in ops:
            for m2 in ops:
                A = np.kron(m1, m2)
                self.check_decomp(A, qd.decompose_4x4_optimal(A), tol=2e-9)

    def test_decompose_4x4_optimal_random_unitary(self):
        np.random.seed(100)
        for _ in range(10):
            A = unitary_group.rvs(4)
            self.check_decomp(A, qd.decompose_4x4_optimal(A))

    def test_decompose_4x4_optimal_random_orthogonal(self):
        np.random.seed(100)
        for _ in range(10):
            A = ortho_group.rvs(4)
            self.check_decomp(A, qd.decompose_4x4_optimal(A))

    def test_decompose_4x4_tp(self):
        np.random.seed(100)
        for _ in range(10):
            U = np.kron(self._random_su(2), self._random_su(2))
            A, B = qd.decompose_4x4_tp(U)
            self.assertAllClose(U, np.kron(A, B))

    def test_matrix_to_cirq_circuit(self):

        def _check(A):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertAllClose(A, qd.matrix_to_cirq_circuit(A).unitary())

        _check(SWAP)
        _check(CNOT)
        _check(QFT_2)

        np.random.seed(100)
        for matrix_size in [2, 4, 8]:
            for _ in range(10):
                _check(ortho_group.rvs(matrix_size))
                _check(unitary_group.rvs(matrix_size))

    def test_decompose_product_state(self):
        def _check(state):
            a, b = decompose_product_state(np.array(state))
            assert np.allclose(np.kron(a, b), state)

        _check([0, -3e-49j, -np.sqrt(0.5), -np.sqrt(0.5)])
        _check([-6.29490599e-09 - 7.85046229e-17j, 0, 0, -1e-09 + 1.00000000e+00j])


if __name__ == '__main__':
    unittest.main()
