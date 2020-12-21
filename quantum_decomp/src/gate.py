import numpy as np

from quantum_decomp.src.two_level_unitary import TwoLevelUnitary


class Gate:
    """Represents gate acting on register of qubits."""
    pass


class GateSingle(Gate):
    """Represents gate acting on a single qubit in a register."""

    def __init__(self, gate2, qubit_id):
        self.gate2 = gate2
        self.qubit_id = qubit_id

    def to_qsharp_command(self, qubits_count):
        if self.gate2.name in ('Rx', 'Ry', 'Rz'):
            # QSharp uses different sign.
            return '%s(%.15f, qs[%d]);' % (
                self.gate2.name, -self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'R1':
            return 'R1(%.15f, qs[%d]);' % (self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'X':
            return 'X(qs[%d]);' % (self.qubit_id)

    def to_matrix(self, qubits_count):
        """Tensor product I x I x ... x `gate2.to_matrix()` x I x ... x I."""
        matrix = self.gate2.to_matrix()
        tile_size = 2 ** (self.qubit_id + 1)
        ts2 = tile_size // 2  # Half tile size.

        if (self.qubit_id == 0):
            tile = matrix
        else:
            tile = np.zeros((tile_size, tile_size), dtype=np.complex128)
            subtile = np.eye(tile_size // 2)
            for i in range(2):
                for j in range(2):
                    tile[i * ts2:(i + 1) * ts2,
                         j * ts2:(j + 1) * ts2] = subtile * matrix[i, j]

        matrix_size = 2 ** qubits_count
        ret = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
        for i in range(2 ** (qubits_count - self.qubit_id - 1)):
            ret[i * tile_size:(i + 1) * tile_size,
                i * tile_size:(i + 1) * tile_size] = tile

        return ret

    def __repr__(self):
        return str(self.gate2) + " on bit " + str(self.qubit_id)

    def type(self):
        return self.gate2.name + "-single"


class GateFC(Gate):
    """ Represents fully contolled gate."""

    def __init__(self, gate2, qubit_id):
        self.gate2 = gate2
        self.qubit_id = qubit_id

    def to_qsharp_command(self, qubits_count):
        # On one qubit controlled gate is just single-qubit gate.
        if qubits_count == 1:
            return GateSingle(self.gate2, self.qubit_id).to_qsharp_command(
                qubits_count)

        control_ids = [i for i in range(qubits_count) if i != self.qubit_id]
        controls = '[' + ', '.join(['qs[%d]' % i for i in control_ids]) + ']'
        if self.gate2.name in ('Rx', 'Ry', 'Rz'):
            # QSharp uses different sign.
            return 'Controlled %s(%s, (%.15f, qs[%d]));' % (
                self.gate2.name, controls, -self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'R1':
            return 'Controlled R1(%s, (%.15f, qs[%d]));' % (
                controls, self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'X':
            if qubits_count == 2:
                return 'CNOT(qs[%d], qs[%d]);' % (
                    control_ids[0], self.qubit_id)
            return 'Controlled X(%s, (qs[%d]));' % (controls, self.qubit_id)

    def to_matrix(self, qubits_count):
        matrix_size = 2 ** qubits_count
        index2 = (matrix_size - 1)
        index1 = index2 - 2 ** self.qubit_id
        matrix = TwoLevelUnitary(
            self.gate2.to_matrix(),
            matrix_size,
            index1,
            index2)
        return matrix.get_full_matrix()

    def __repr__(self):
        return "%s on bit %d, fully controlled" % (
            str(self.gate2), self.qubit_id)

    def type(self):
        return self.gate2.name + "-FC"


def gates_to_matrix(gates, qubits_count):
    """Converts gate sequence to matrix implemented by this sequence."""
    result = np.eye(2 ** qubits_count)
    for gate in gates:
        assert isinstance(gate, Gate)
        result = gate.to_matrix(qubits_count) @ result
    return result


def apply_on_qubit(gates, qubit_id):
    """Converts Gate2 gates to GateSingle gates acting on the same qubit."""
    return [GateSingle(gate, qubit_id) for gate in gates]
