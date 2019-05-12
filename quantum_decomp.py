import numpy as np
import math

PAULI_X = np.array([[0, 1], [1, 0]])


def check_unitary(A):
    n = A.shape[0]
    if (A.shape != (n, n)):
        raise ValueError("Matrix is not square.")
    A = np.array(A)
    if (np.linalg.norm(np.eye(n) - A @ A.conj().T) > 1e-9):
        raise ValueError("Matrix is not unitary: %s" % str(A))


def check_special_unitary(A):
    check_unitary(A)
    assert np.abs(np.linalg.det(A) - 1.0) < 1e-9


def check_power_of_two(x):
    if 2**int(math.log2(x)) != x:
        raise ValueError(str(x) + " is not a power of two.")

# Represents two-lebel unitary matrix, i.e. a unitary matrix obtained from the identity
# matrix by changing a 2x2 principal submatrix.


class TwoLevelUnitary:
    def __init__(self, matrix2x2, matrix_size, index1, index2):
        assert index1 != index2
        assert index1 < matrix_size and index2 < matrix_size
        assert matrix2x2.shape == (2, 2)
        check_unitary(matrix2x2)

        self.matrix_size = matrix_size
        self.index1 = index1
        self.index2 = index2
        self.matrix_2x2 = matrix2x2

    def __repr__(self):
        return "%s on (%d, %d)" % (
            str(self.matrix_2x2), self.index1, self.index2)

    def order_indices(self):
        if self.index1 > self.index2:
            self.index1, self.index2 = self.index2, self.index1
            self.matrix_2x2 = PAULI_X @ self.matrix_2x2 @ PAULI_X

    def get_full_matrix(self):
        matrix_full = np.array(np.eye(self.matrix_size, dtype=np.complex128))
        matrix_full[self.index1, self.index1] = self.matrix_2x2[0, 0]
        matrix_full[self.index1, self.index2] = self.matrix_2x2[0, 1]
        matrix_full[self.index2, self.index1] = self.matrix_2x2[1, 0]
        matrix_full[self.index2, self.index2] = self.matrix_2x2[1, 1]
        return matrix_full

    def apply_permutation(self, perm):
        assert(len(perm) == self.matrix_size)
        self.index1 = perm[self.index1]
        self.index2 = perm[self.index2]

    # Returns list of fully controlled gates implementing this matrix.
    def to_fc_gates(self):
        self.order_indices()
        qubit_id_mask = self.index1 ^ self.index2
        check_power_of_two(qubit_id_mask)
        assert self.index1 < self.index2

        qubit_id = int(math.log2(qubit_id_mask))
        flip_mask = (self.matrix_size - 1) - self.index2
        qubit_count = int(math.log2(self.matrix_size))

        return [GateFC(gate2, qubit_id, qubit_count, flip_mask=flip_mask)
                for gate2 in unitary2x2_to_gates(self.matrix_2x2)]


# Returns list of two-level unitary matrices, which multiply to A.
# Matrices are listed in application order, i.e. if aswer is [u_1, u_2,
# u_3], it means A = u_3 u_2 u_1.
def two_level_decompose(A):
    # Returns unitary matrix U, s.t. [a, b] U = [c, 0].
    # makes second element equal to zero.
    def make_eliminating_matrix(a, b):
        assert (np.abs(a) > 1e-9 and np.abs(b) > 1e-9)
        theta = np.arctan(np.abs(b / a))
        lmbda = -np.angle(a)
        mu = np.pi + np.angle(b) - np.angle(a) - lmbda
        result = np.array([[np.cos(theta) * np.exp(1j * lmbda),
                            np.sin(theta) * np.exp(1j * mu)],
                           [-np.sin(theta) * np.exp(-1j * mu),
                            np.cos(theta) * np.exp(-1j * lmbda)]])
        check_special_unitary(result)
        assert np.allclose(np.angle(result[0, 0] * a + result[1, 0] * b), 0)
        assert (np.abs(result[0, 1] * a + result[1, 1] * b) < 1e-9)
        return result

    check_unitary(A)
    n = A.shape[0]
    result = []
    current_A = A

    for i in range(n - 2):
        for j in range(n - 1, i, -1):
            if abs(current_A[i, j]) < 1e-9:
                # Element is already zero, skipping.
                pass
            else:
                if abs(current_A[i, j - 1]) < 1e-9:
                    # Just swap columns.
                    u_2x2 = np.array([[0, 1], [1, 0]])
                else:
                    u_2x2 = make_eliminating_matrix(
                        current_A[i, j - 1], current_A[i, j])
                check_unitary(u_2x2)
                current_A = current_A @ TwoLevelUnitary(
                    u_2x2, n, j - 1, j).get_full_matrix()
                u_2x2_inv = u_2x2.conj().T
                result.append(TwoLevelUnitary(u_2x2_inv, n, j - 1, j))
                assert(np.abs(current_A[i, j]) < 1e-9)

    result.append(TwoLevelUnitary(
        current_A[n - 2:n, n - 2:n], n, n - 2, n - 1))
    return result


def two_level_decompose_gray(A):
    """Retunrs list of two-level matrices, which multiplu to A.
    Guarantees that each matrix acts on single bit."""
    N = A.shape[0]
    check_power_of_two(N)
    assert A.shape == (N, N), "Matrix must be square."
    check_unitary(A)

    # Build permutation matrix.
    perm = [x ^ (x // 2) for x in range(N)]  # Gray code.
    P = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        P[i][perm[i]] = 1

    result = two_level_decompose(P @ A @ P.T)
    for matrix in result:
        matrix.apply_permutation(perm)
    return result


def su_to_gates(A):
    """Decomposes 2x2 special unitary to gates Ry, Rz.
    R_k(x) = exp(0.5*i*x*sigma_k)."""
    check_special_unitary(A)
    u00 = A[0, 0]
    u01 = A[0, 1]
    theta = np.arccos(np.abs(u00))
    lmbda = np.angle(u00)
    mu = np.angle(u01)

    result = []
    result.append(Gate2('Rz', lmbda - mu))
    result.append(Gate2('Ry', 2 * theta))
    result.append(Gate2('Rz', lmbda + mu))
    return result


# Decomposes 2x2 unitary to gates Ry, Rz, R1.
# R1(x) = diag(1, exp(i*x)).
def unitary2x2_to_gates(A):
    check_unitary(A)
    phi = np.angle(np.linalg.det(A))
    if np.abs(phi) < 1e-9:
        return su_to_gates(A)
    else:
        A = np.diag([1.0, np.exp(-1j * phi)]) @ A
        return su_to_gates(A) + [Gate2('R1', phi)]


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

# Represents gate acting on register of qubits.


class Gate:
    pass


class GateSingle(Gate):
    """Represents gate acting on a single qubit in a register."""

    def __init__(self, gate2, qubit_id, qubit_count):
        self.gate2 = gate2
        self.qubit_id = qubit_id
        self.qubit_count = qubit_count

    def to_qsharp_command(self):
        if self.gate2.name in ('Rx', 'Ry', 'Rz'):
            # QSharp uses different sign.
            return '%s(%.15f, qs[%d]);' % (
                self.gate2.name, -self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'R1':
            return 'R1(%.15f, qs[%d]);' % (self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'X':
            return 'X(qs[%d]);' % (self.qubit_id)

    def to_matrix(self):
        """Tensor product I x I x ... x `gate2.to_matrix()` x I x ... x I."""
        matrix = self.gate2.to_matrix()
        tile_size = 2**(self.qubit_id + 1)

        if (self.qubit_id == 0):
            tile = matrix
        else:
            tile = np.zeros((tile_size, tile_size), dtype=np.complex128)
            subtile = np.eye(tile_size // 2)
            for i in range(2):
                for j in range(2):
                    tile[i * (tile_size // 2):(i + 1) * (tile_size // 2), j *
                         (tile_size // 2):(j + 1) * (tile_size // 2)] = subtile * matrix[i, j]

        matrix_size = 2 ** self.qubit_count
        ret = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
        for i in range(2**(self.qubit_count - self.qubit_id - 1)):
            ret[i * tile_size:(i + 1) * tile_size,
                i * tile_size:(i + 1) * tile_size] = tile

        return ret

    def __repr__(self):
        return str(self.gate2) + " on bit " + str(self.qubit_id)


class GateFC(Gate):
    """ Represents fully contolled gate.
    `flip_mask` has ones at positions, for which qubit should be flipped before and after
    applying operation.
    """

    def __init__(self, gate2, qubit_id, qubit_count, flip_mask=0):
        self.gate2 = gate2
        self.qubit_id = qubit_id
        self.flip_mask = flip_mask
        self.qubit_count = qubit_count

    def without_flips(self):
        return GateFC(self.gate2, self.qubit_id, self.qubit_count, flip_mask=0)

    def to_qsharp_command(self):
        # On one qubit controlled gate is just single-qubit gate.
        if self.qubit_count == 1:
            return GateSingle(self.gate2, self.qubit_id, 1).to_qsharp_command()

        if self.flip_mask != 0:
            raise ValueError("flip_mask must be zero.")

        controls = [i for i in range(self.qubit_count) if i != self.qubit_id]
        controls = '[' + ', '.join(['qs[%d]' % i for i in controls]) + ']'
        if self.gate2.name in ('Rx', 'Ry', 'Rz'):
            # QSharp uses different sign.
            return 'Controlled %s(%s, (%.15f, qs[%d]));' % (
                self.gate2.name, controls, -self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'R1':
            return 'Controlled R1(%s, (%.15f, qs[%d]));' % (
                controls, self.gate2.arg, self.qubit_id)
        elif self.gate2.name == 'X':
            return 'Controled X(%s, (qs[%d]));' % (controls, self.qubit_id)

    def to_matrix(self):
        matrix_size = 2**self.qubit_count
        index2 = (matrix_size - 1) - self.flip_mask
        index1 = index2 - 2**self.qubit_id
        matrix = TwoLevelUnitary(
            self.gate2.to_matrix(),
            matrix_size,
            index1,
            index2)
        return matrix.get_full_matrix()

    def __repr__(self):
        return "%s on bit %d, fully controlled" % (
            str(self.gate2), self.qubit_id)


def optimize_gates(gates):
    """Cancels consequent NOT gates. Skips identity gates.
    After execution all fully controlled gates will have `flip_mask=0`."""
    qubit_count = gates[0].qubit_count
    for gate in gates:
        assert gate.qubit_count == qubit_count

    result = []

    global flip_mask
    flip_mask = 0

    def dump_flips():
        global flip_mask
        for qubit_id in range(qubit_count):
            if (flip_mask & (2**qubit_id)) != 0:
                result.append(GateSingle(Gate2('X'), qubit_id, qubit_count))
        flip_mask = 0

    for gate in gates:
        if isinstance(gate, GateSingle):
            if gate.gate2.name == 'X':
                flip_mask ^= 2**gate.qubit_id
            elif gate.gate2.is_identity():
                pass
            else:
                dump_flips()
                result.append(gate)
        else:
            assert isinstance(gate, GateFC)
            if gate.gate2.is_identity():
                pass
            else:
                flip_mask ^= gate.flip_mask
                dump_flips()
                result.append(gate.without_flips())
                flip_mask ^= gate.flip_mask
    dump_flips()

    return result


def matrix_to_gates(A):
    """Given unitary matrix A, retuns sequence of gates which implements
    action of this matrix on register of qubits.

    Input: A - 2^n x 2^N unitary matrix.
    Returns: sequence of Gate objects.
    """
    matrices = two_level_decompose_gray(A)
    gates = sum([matrix.to_fc_gates() for matrix in matrices], [])
    gates = optimize_gates(gates)
    return gates


def gates_to_matrix(gates):
    """Given sequence of gates, returns a unitary matrix which implemented by it."""
    result = np.eye(2 ** gates[0].qubit_count)
    for gate in gates:
        assert isinstance(gate, Gate)
        result = gate.to_matrix() @ result
    return result


def matrix_to_qsharp(A):
    """Given unitary matrix A, retuns Q# code which implements
    action of this matrix on register of qubits called `qs`.

    Input: A - 2^N x 2^N unitary matrix.
    Returns: string - Q# code.
    """
    header = "operation ApplyUnitaryMatrix (qs : Qubit[]) : Unit {\nbody (...) {\n"
    footer = "  }\n}\n"
    code = '\n'.join(['    ' + gate.to_qsharp_command()
                      for gate in matrix_to_gates(A)])
    return header + code + '\n' + footer


def gates_to_qasm(gates, file_name):
    """Generates qasm code describing a circuit made of given gates."""
    qubit_count = gates[0].qubit_count
    qubit_def = '\n'.join(["qubit\tq%d" % i for i in range(qubit_count)])
    gate_def = ''
    gate_list = ''
    gate_id = 0

    for gate in gates:
        gate_name = str(gate.gate2)
        if isinstance(gate, GateSingle):
            gate_def += "def\tg%d,0,'%s'\n" % (gate_id, gate_name)
            gate_list += "g%d\tq%d\n" % (gate_id, gate.qubit_id)
        else:
            assert isinstance(gate, GateFC)
            assert gate.flip_mask == 0
            qubits_list = [i for i in range(
                qubit_count) if i != gate.qubit_id] + [gate.qubit_id]
            qubits_list = ','.join(['q%d' % i for i in qubits_list])
            gate_def += "def\tg%d,%d,'%s'\n" % (gate_id,
                                                qubit_count - 1, gate_name)
            gate_list += "g%d\t%s\n" % (gate_id, qubits_list)
        gate_id += 1

    qasm_code = qubit_def + '\n\n' + gate_def + '\n' + gate_list
    with open(file_name, 'w') as f:
        f.write(qasm_code)
