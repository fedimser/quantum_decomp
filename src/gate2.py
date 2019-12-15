import numpy as np

from src.utils import PAULI_X


class Gate2:
    """Represents gate acting on one qubit."""

    def __init__(self, name, arg=None):
        assert name in ['Ry', 'Rz', 'R1', 'X']
        self.name = name
        self.arg = arg

    def to_matrix(self):
        if self.name == 'Ry':
            return np.array([[np.cos(self.arg / 2), np.sin(self.arg / 2)],
                             [-np.sin(self.arg / 2), np.cos(self.arg / 2)]])
        elif self.name == 'Rz':
            return np.diag([np.exp(0.5j * self.arg), np.exp(-0.5j * self.arg)])
        elif self.name == 'R1':
            return np.diag([1.0, np.exp(1j * self.arg)])
        elif self.name == 'X':
            return PAULI_X

    def is_identity(self):
        return np.linalg.norm(self.to_matrix() - np.eye(2)) < 1e-10

    def __repr__(self):
        if self.arg is not None:
            return self.name + "(" + str(self.arg) + ")"
        else:
            return self.name
