import numpy as np

from quantum_decomp.src.gate import GateSingle, GateFC
from quantum_decomp.src.gate2 import Gate2


def test_GateSingle_to_matrix():
    assert np.allclose(GateSingle(Gate2('X'), 0, 2).to_matrix(),
                       [[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

    assert np.allclose(GateSingle(Gate2('X'), 1, 2).to_matrix(),
                       [[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]])


def test_GateFC_to_matrix():
    assert np.allclose(GateFC(Gate2('X'), 0, 2).to_matrix(),
                       [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

    assert np.allclose(GateFC(Gate2('X'), 1, 2).to_matrix(),
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]])
