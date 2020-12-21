import numpy as np

from quantum_decomp.src.gate import GateSingle, GateFC
from quantum_decomp.src.gate2 import Gate2


def test_GateSingle_to_matrix():
    assert np.allclose(GateSingle(Gate2('X'), 0).to_matrix(2),
                       [[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

    assert np.allclose(GateSingle(Gate2('X'), 1).to_matrix(2),
                       [[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]])


def test_GateFC_to_matrix():
    assert np.allclose(GateFC(Gate2('X'), 0).to_matrix(2),
                       [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

    assert np.allclose(GateFC(Gate2('X'), 1).to_matrix(2),
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]])
