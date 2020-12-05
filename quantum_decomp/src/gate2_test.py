import numpy as np

from quantum_decomp.src.gate2 import Gate2


def test_Gate2_to_matrix():
    assert np.allclose(Gate2('X').to_matrix(), [[0, 1], [1, 0]])

    assert np.allclose(Gate2('R1', 0).to_matrix(), np.diag([1, 1]))
    assert np.allclose(
        Gate2('R1', np.pi / 2).to_matrix(), np.diag([1, 1j]))
    assert np.allclose(Gate2('R1', np.pi).to_matrix(), np.diag([1, -1]))

    assert np.allclose(Gate2('Ry', 0).to_matrix(), np.diag([1, 1]))
    assert np.allclose(Gate2('Ry', np.pi / 2).to_matrix(),
                       np.array([[1, 1], [-1, 1]]) / np.sqrt(2))
    assert np.allclose(Gate2('Ry', np.pi).to_matrix(),
                       np.array([[0, 1], [-1, 0]]))

    assert np.allclose(Gate2('Rz', 0).to_matrix(), np.diag([1, 1]))
    assert np.allclose(
        Gate2('Rz', np.pi).to_matrix(), np.diag([1j, -1j]))
