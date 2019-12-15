"""
Optimal decomposition of general 4x4 unitary into single-qubit gates and
at most 3 CNOT gates.

Referecnes:
[1] Farrokh Vatan, Colin Williams.
    Optimal Quantum Circuits for General Two-Qubit Gates.
    https://arxiv.org/pdf/quant-ph/0308006.pdf
[2] B. Kraus and J.I. Cirac.
    Optimal Creation of Entanglement Using a Two-Qubit Gate.
    https://arxiv.org/pdf/quant-ph/0011050.pdf
"""

import numpy as np

from src.gate import GateSingle, GateFC, apply_on_qubit, gates_to_matrix
from src.gate2 import Gate2
from src.decompose_2x2 import su_to_gates
from src.optimize import optimize_gates
from src.utils import cast_to_real, is_unitary


# "Magic basis". Columns are Phi vectors defined in [2].
# Last two columns replaced to make formula A2 true.
# Columns of Phi form maximally entangled basis.
Phi = np.sqrt(0.5) * np.array([[1, -1j, 0, 0],
                               [0, 0, -1j, 1],
                               [0, 0, -1j, -1],
                               [1, 1j, 0, 0]])
Phi_dag = Phi.conj().T


def _allclose(x, y):
    return np.allclose(x, y, atol=1e-7)


def magic_N(a):
    """Builds 'Magic N' matrix.

    Magic N matrix is 4x4 non-local unitary defined by 3 real parameters, s.t.:
    N(a_x, a_y, a_z)= exp[i(a_x*S_x + a_y*S_y + a_x*S_z)], where
    S_k = sigma_k_A x sigma_k_B (tensor product of Pauli matrices).

    Magic N matrix is diagonal in "magic basis".
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    I4 = np.eye(4)
    N1 = np.cos(a[0]) * I4 + 1j * np.sin(a[0]) * np.kron(X, X)
    N2 = np.cos(a[1]) * I4 + 1j * np.sin(a[1]) * np.kron(Y, Y)
    N3 = np.cos(a[2]) * I4 + 1j * np.sin(a[2]) * np.kron(Z, Z)
    return N1 @ N2 @ N3


def trace_B(A):
    """Partial trace."""
    assert A.shape == (4, 4)
    return np.array([[A[0, 0] + A[1, 1], A[0, 2] + A[1, 3]],
                     [A[2, 0] + A[3, 1], A[2, 2] + A[3, 3]]])


def is_maximally_entangled_state(x):
    """Checks if state of 2 qubits is maximally entangled.

    A maximally entangled state is a quantum state which has maximum von
    Neumann entropy for each bipartition.
    """
    assert x.shape == (4, )
    assert np.allclose(np.dot(x, x.conj()), 1)
    rho = np.outer(x.conj(), x)
    rho_a = trace_B(rho)
    eigv, _ = np.linalg.eig(rho_a)
    entr = -sum([y * np.log2(y) for y in eigv if y > 0])
    return np.allclose(entr, 1)


def is_maximally_entangled_basis(A):
    """
    Checks if columns of matrx form a maximally entanfgled basis.

    Maximally entanfgled basis is basis consisting of maximally entangled
    orthonormal states.
    """
    assert A.shape == (4, 4)
    assert is_unitary(A)
    return all([is_maximally_entangled_state(A[:, i]) for i in range(4)])


def to_real_in_magic_basis(state):
    """By only changing global phase, makes `state` real in magic basis."""
    assert is_maximally_entangled_state(state)
    state2 = Phi_dag @ state
    y = 0
    for i in range(4):
        if np.abs(state2[i]) > 1e-9:
            y = -np.angle(state2[i])
    result = np.exp(1j * y) * state
    assert np.allclose(Phi @ cast_to_real(Phi_dag @ result), result)
    return result


def decompose_product_state(state):
    """Decomposes product state.

    Given state which is tensor product of two qubit states, returns these
    states.
    Throws AssertionError if state is not product state.
    """
    assert state.shape == (4,)
    assert np.allclose(np.linalg.norm(state), 1)

    def normalize(x, y, alt_x, alt_y):
        norm = np.sqrt(x**2 + y**2)
        if norm < 1e-9:
            norm = np.sqrt(alt_x**2 + alt_y**2)
            assert(norm > 1e-9)
            return alt_x / norm, alt_y / norm
        else:
            return x / norm, y / norm

    c = np.abs(state)
    phase = np.angle(state)
    a1, a2 = normalize(c[0], c[2], c[1], c[3])
    b1, b2 = normalize(c[0], c[1], c[2], c[3])

    a = np.array([a1, a2 * np.exp(1j * (phase[2] - phase[0]))])
    b = np.array([b1 * np.exp(1j * phase[0]), b2 * np.exp(1j * phase[1])])

    assert np.allclose(np.kron(a, b), state)
    return a, b


def decompose_4x4_partial(Psi):
    """Partially decomposes 4x4 unitary matrix.

    Takes matrix Psi, columns of which form fully entangled basis.
    Returns matrices UA, UB and vector zeta, such that:
    (UAxUB) * Psi * exp(i * diag(zeta)) = Phi.

    Implements algorithm in Lemma 1 in Appendix A from [2].
    """
    assert is_maximally_entangled_basis(Psi)

    Psi_bar = np.zeros_like(Psi)
    for i in range(4):
        Psi_bar[:, i] = to_real_in_magic_basis(Psi[:, i])

    e_f = (Psi_bar[:, 0] + 1j * Psi_bar[:, 1]) / np.sqrt(2)
    e_ort_f_ort = (Psi_bar[:, 0] - 1j * Psi_bar[:, 1]) / np.sqrt(2)
    e, f = decompose_product_state(e_f)
    e_ort, f_ort = decompose_product_state(e_ort_f_ort)
    assert np.allclose(np.dot(e.conj(), e_ort), 0)
    assert np.allclose(np.dot(f.conj(), f_ort), 0)

    def restore_phase(a, b, c):
        """Finds such delta, that a*exp(i*delta) + b*exp(-i*delta)=c."""
        assert np.abs(a) >= 1e-9
        x1 = (c + np.sqrt(c * c - 4 * a * b)) / (2 * a)
        x2 = (c - np.sqrt(c * c - 4 * a * b)) / (2 * a)
        x = x1
        if not np.allclose(np.abs(x), 1):
            x = x2
        assert np.allclose(np.abs(x), 1)
        delta = np.angle(x)
        assert np.allclose(a * np.exp(1j * delta) + b * np.exp(-1j * delta), c)
        return delta

    a_d = np.kron(e, f_ort)
    b_d = np.kron(e_ort, f)
    c_d = np.sqrt(2) * 1j * Psi_bar[:, 2]
    i_d = 0
    while np.abs(a_d[i_d]) < 1e-9:
        i_d += 1
    delta = restore_phase(a_d[i_d], b_d[i_d], c_d[i_d])

    e_f_ort = np.kron(e, f_ort)
    e_ort_f = np.kron(e_ort, f)

    # Correcting ambiguity in sign.
    p3_1 = Psi_bar[:, 3]
    p3_2 = (e_f_ort * np.exp(1j * delta) - e_ort_f *
            np.exp(-1j * delta)) / np.sqrt(2)
    if any([np.real(p3_1[i] / p3_2[i]) < -0.5 for i in range(4)]):
        Psi_bar[:, 3] = -Psi_bar[:, 3]

    # Check formulas A3-A5.
    assert is_unitary(np.array([e_f, e_f_ort, e_ort_f, e_ort_f_ort]))
    assert np.allclose(Psi_bar[:, 0], (e_f + e_ort_f_ort) / np.sqrt(2))
    assert np.allclose(Psi_bar[:, 1], -1j * (e_f - e_ort_f_ort) / np.sqrt(2))
    p2 = -1j * (e_f_ort * np.exp(1j * delta) + e_ort_f *
                np.exp(-1j * delta)) / np.sqrt(2)
    assert np.allclose(Psi_bar[:, 2], p2)
    p3 = (e_f_ort * np.exp(1j * delta) - e_ort_f *
          np.exp(-1j * delta)) / np.sqrt(2)
    assert _allclose(Psi_bar[:, 3], p3)

    UA = np.array([e.conj(), e_ort.conj() * np.exp(1j * delta)])
    UB = np.array([f.conj(), f_ort.conj() * np.exp(-1j * delta)])
    assert is_unitary(UA)
    assert is_unitary(UB)

    UAUB = np.kron(UA, UB)
    UAUB_alt = np.array([e_f.conj(),
                         e_f_ort.conj() * np.exp(-1j * delta),
                         e_ort_f.conj() * np.exp(1j * delta),
                         e_ort_f_ort.conj()])
    assert np.allclose(UAUB, UAUB_alt)
    assert is_unitary(UAUB)

    assert np.allclose(UAUB @ Psi_bar, Phi)

    D = (UAUB @ Psi).conj().T @ Phi
    zeta = np.angle(np.array([D[i, i] for i in range(4)]))
    assert _allclose(D, np.diag(np.exp(1j * zeta)))

    assert np.allclose(np.kron(UA, UB) @ Psi @ np.diag(np.exp(1j * zeta)), Phi)
    return UA, UB, zeta


def decompose_to_magic_diagonal(U):
    """Decomposes unitary U as follows:
    U = e^(if) * (UAxUB) * N(alpha) * (VAxVB)
    f - global phase.
    UA, UB, VA, VB - 1-qubit _special_ unitaries (det=1).
    N(alpha) - "Magic N" matrix.

    Implements algorithm described in Appendix A in [2].
    """
    # Step 1 in paper.
    UT = Phi @ (Phi_dag @ U @ Phi).T @ Phi_dag
    eig_values, eig_vecs = np.linalg.eig(UT @ U)
    Psi = eig_vecs
    assert is_maximally_entangled_basis(Psi)

    eps = cast_to_real(np.log(eig_values) / (2 * 1j))

    # Step 2 in paper.
    VA, VB, xi = decompose_4x4_partial(Psi)

    # Step 3 in paper.
    Psi_tilde = U @ Psi @ np.diag(np.exp(-1j * eps))
    assert is_maximally_entangled_basis(Psi_tilde)

    # Step 4 in paper.
    UA_dag, UB_dag, lxe = decompose_4x4_partial(Psi_tilde)
    lmbda = lxe - xi - eps
    UA = UA_dag.conj().T
    UB = UB_dag.conj().T

    UAUB = np.kron(UA, UB)
    VAVB = np.kron(VA, VB)

    Ud = Phi @ np.diag(np.exp(-1j * lmbda)) @ Phi.conj().T

    assert np.allclose(VAVB @ Psi @ np.diag(np.exp(1j * xi)), Phi)
    assert np.allclose(U @ Psi @ np.diag(np.exp(-1j * eps)), Psi_tilde)
    assert np.allclose(UAUB.conj().T @ Psi_tilde @ np.diag(
        np.exp(1j * (eps + xi + lmbda))), Phi)
    assert _allclose(U, np.kron(UA, UB) @ Ud @ np.kron(VA, VB))

    # Restore coefficients of non-local unitary.
    gl_phase = - 0.25 * np.sum(lmbda)
    alpha_x = -0.5 * (lmbda[0] + lmbda[2] + 2 * gl_phase)
    alpha_y = -0.5 * (lmbda[1] + lmbda[2] + 2 * gl_phase)
    alpha_z = -0.5 * (lmbda[0] + lmbda[1] + 2 * gl_phase)
    alpha = np.array([alpha_x, alpha_y, alpha_z])
    assert np.allclose(lmbda[0], - gl_phase - alpha_x + alpha_y - alpha_z)
    assert np.allclose(lmbda[1], - gl_phase + alpha_x - alpha_y - alpha_z)
    assert np.allclose(lmbda[2], - gl_phase - alpha_x - alpha_y + alpha_z)
    assert np.allclose(lmbda[3], - gl_phase + alpha_x + alpha_y + alpha_z)

    assert _allclose(Ud, np.exp(1j * gl_phase) * magic_N(alpha))
    assert _allclose(U, np.exp(1j * gl_phase) * UAUB @ magic_N(alpha) @ VAVB)

    def correct_phase(A, cur_phase):
        """Correct phases of local unitaries to make them speical unitaries."""
        assert is_unitary(A)
        f = 0.5 * np.angle(np.linalg.det(A))
        return A * np.exp(-1j * f), cur_phase + f

    UA, gl_phase = correct_phase(UA, gl_phase)
    UB, gl_phase = correct_phase(UB, gl_phase)
    VA, gl_phase = correct_phase(VA, gl_phase)
    VB, gl_phase = correct_phase(VB, gl_phase)

    for mx in [UA, UB, VA, VB]:
        assert np.allclose(np.linalg.det(mx), 1)

    UAUB = np.kron(UA, UB)
    VAVB = np.kron(VA, VB)
    assert _allclose(U, np.exp(1j * gl_phase) * UAUB @ magic_N(alpha) @ VAVB)

    return {
        'UA': UA,
        'UB': UB,
        'VA': VA,
        'VB': VB,
        'alpha': alpha,
        'global_phase': gl_phase,
    }


def decompose_magic_N(a):
    """Decomposes "Magic N" matrix into 3 CNOTs, 4 Rz and 1 Ry gate.

    Result is missing global phase pi/4.
    Implements cirquit on fig. 7 from [1].
    """
    t1 = 2 * a[2] - 0.5 * np.pi
    t2 = 0.5 * np.pi - 2 * a[0]
    t3 = 2 * a[1] - 0.5 * np.pi
    result = []

    result.append(GateSingle(Gate2('Rz', 0.5 * np.pi), 1, 2))
    result.append(GateFC(Gate2('X'), 0, 2))
    result.append(GateSingle(Gate2('Rz', t1), 0, 2))
    result.append(GateSingle(Gate2('Ry', t2), 1, 2))
    result.append(GateFC(Gate2('X'), 1, 2))
    result.append(GateSingle(Gate2('Ry', t3), 1, 2))
    result.append(GateFC(Gate2('X'), 0, 2))
    result.append(GateSingle(Gate2('Rz', -0.5 * np.pi), 0, 2))

    N = magic_N(a)
    assert np.allclose(N, gates_to_matrix(result) * np.exp(0.25j * np.pi))

    return result


def decompose_4x4_optimal(U):
    """Builds optimal decomposition of general 4x4 unitary matrix.

    This decomposition consists of at most 3 CNOT gates, 15 Rx/Ry gates and one
    R1 gate.
    """
    magic_decomp = decompose_to_magic_diagonal(U)

    result = []
    result += apply_on_qubit(su_to_gates(magic_decomp['VA']), 1, 2)
    result += apply_on_qubit(su_to_gates(magic_decomp['VB']), 0, 2)
    result += decompose_magic_N(magic_decomp['alpha'])
    result += apply_on_qubit(su_to_gates(magic_decomp['UA']), 1, 2)
    result += apply_on_qubit(su_to_gates(magic_decomp['UB']), 0, 2)

    # Adding global phase using Rz and R1.
    gl_phase = magic_decomp['global_phase'] + 0.25 * np.pi
    if np.abs(gl_phase) > 1e-9:
        result.append(GateSingle(Gate2('Rz', 2 * gl_phase), 0, 2))
        result.append(GateSingle(Gate2('R1', 2 * gl_phase), 0, 2))

    result = optimize_gates(result)

    assert _allclose(U, gates_to_matrix(result))

    return result
