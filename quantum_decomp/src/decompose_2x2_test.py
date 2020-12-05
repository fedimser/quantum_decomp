import numpy as np

from quantum_decomp.src.decompose_2x2 import unitary2x2_to_gates
from quantum_decomp.src.test_utils import random_unitary


def test_unitary2x2_to_gates():
    for i in range(100):
        A = np.array(random_unitary(2))
        gates = unitary2x2_to_gates(A)
        assert len(gates) <= 4

        B = np.eye(2)
        for gate in gates:
            B = gate.to_matrix() @ B
        assert np.linalg.norm(A - B) < 1e-9
