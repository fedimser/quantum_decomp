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

from quantum_decomp.src.gate import (GateSingle, GateFC, apply_on_qubit,
                                     gates_to_matrix)
from quantum_decomp.src.gate2 import Gate2
from quantum_decomp.src.decompose_2x2 import su_to_gates
from quantum_decomp.src.linalg import orthonormal_eigensystem
from quantum_decomp.src.utils import (
    cast_to_real, is_real, is_special_unitary, is_unitary, skip_identities)

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


def correct_phase(A, cur_phase):
    """Correct phases of 2x2 unitary to make it speical unitary."""
    assert is_unitary(A)
    n = A.shape[0]
    f = np.angle(np.linalg.det(A)) / n
    return A * np.exp(-1j * f), cur_phase + f


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
    assert x.shape == (4,)
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

    # c = [a1*b1, a1*b2, a2*b1, a2*b2]
    c = np.abs(state)
    phase = np.angle(state)

    def normalize(x, y):
        ans = np.array([x, y], dtype=np.complex128)
        return ans / np.linalg.norm(ans)

    # Special cases - when one of state is unit vector.
    eps = 1e-8
    if c[0] < eps and c[1] < eps:
        a = np.array([0, 1], dtype=np.complex128)
        b = normalize(state[2], state[3])
    elif c[2] < eps and c[3] < eps:
        a = np.array([1, 0], dtype=np.complex128)
        b = normalize(state[0], state[1])
    elif c[0] < eps and c[2] < eps:
        a = normalize(state[1], state[3])
        b = np.array([0, 1], dtype=np.complex128)
    elif c[1] < eps and c[3] < eps:
        a = normalize(state[0], state[2])
        b = np.array([1, 0], dtype=np.complex128)
    else:
        a = normalize(c[0], c[2])
        b = normalize(c[0], c[1])
        a2_phase = (phase[2] - phase[0])
        if np.abs(c[0]) + np.abs(c[2]) < 1e-9:
            a2_phase = (phase[3] - phase[1])
        a = np.array([a[0], a[1] * np.exp(1j * a2_phase)])
        b = np.array([b[0] * np.exp(1j * phase[0]),
                      b[1] * np.exp(1j * phase[1])])

    assert np.allclose(np.kron(a, b), state)
    return a, b


def decompose_4x4_tp(U):
    """Decomposes 4x4 special unitary which is tensor product.

    Given special unitary matrix which is tensor product of two special unitary
    matrices, returns these matrices.

    Throws AssertionError if such decomposition is impossible.
    """
    assert U.shape == (4, 4)
    assert is_special_unitary(U)
    grid = [(0, 0), (0, 1), (1, 0), (1, 1)]

    B = None
    for x, y in grid:
        B = U[2 * x:2 * x + 2, 2 * y:2 * y + 2]
        det = np.linalg.det(B)
        if np.abs(np.linalg.det(B)) > 1e-9:
            B = B / np.sqrt(det)
            break
    assert is_special_unitary(B)

    x2, y2 = 0, 0
    for x, y in grid:
        if np.abs(B[x, y]) > 1e-9:
            x2, y2 = x, y
    b = B[x2, y2]
    A = np.array([[U[x2, y2] / b, U[x2, y2 + 2] / b],
                  [U[x2 + 2, y2] / b, U[x2 + 2, y2 + 2] / b]])
    A /= np.sqrt(np.linalg.det(A))
    assert is_special_unitary(A)

    assert np.allclose(np.kron(A, B), U)
    return A, B


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
    e_f_ort = np.kron(e, f_ort)
    e_ort_f = np.kron(e_ort, f)
    assert np.allclose(np.dot(e.conj(), e_ort), 0)
    assert np.allclose(np.dot(f.conj(), f_ort), 0)

    def restore_phase(a, b, c):
        """Finds such delta, that a*exp(i*delta) + b*exp(-i*delta)=c."""
        assert np.abs(a) >= 1e-9
        x1 = (c + np.sqrt(c * c - 4 * a * b)) / (2 * a)
        x2 = (c - np.sqrt(c * c - 4 * a * b)) / (2 * a)
        xs = [v for v in [x1, x2] if np.allclose(np.abs(v), 1)]
        deltas = [np.angle(x) for x in xs]
        assert len(deltas) > 0
        for d in deltas:
            assert np.allclose(a * np.exp(1j * d) + b * np.exp(-1j * d), c)
        return deltas

    a_d = np.kron(e, f_ort)
    b_d = np.kron(e_ort, f)
    c_d = np.sqrt(2) * 1j * Psi_bar[:, 2]
    i_d = 0
    while np.abs(a_d[i_d]) < 1e-9:
        i_d += 1
    deltas = restore_phase(a_d[i_d], b_d[i_d], c_d[i_d])

    # If there are 2 solutions for delta, we need to choose correct one.
    delta = deltas[0]
    if len(deltas) == 2:
        delta = None
        for d in deltas:
            p2 = -1j * (e_f_ort * np.exp(1j * d) + e_ort_f *
                        np.exp(-1j * d)) / np.sqrt(2)
            if _allclose(p2, Psi_bar[:, 2]):
                delta = d
        assert delta is not None

    # Correcting ambiguity in sign.
    # Formula A5b has "+/-", and we need to choose correct sign.
    p3_1 = Psi_bar[:, 3]
    p3_2 = (e_f_ort * np.exp(1j * delta) - e_ort_f *
            np.exp(-1j * delta)) / np.sqrt(2)
    negate_k = 1
    for i in range(4):
        if abs(p3_2[i]) > 1e-9 and np.real(p3_1[i] / p3_2[i]) < -0.5:
            negate_k = -1
    Psi_bar[:, 3] *= negate_k

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
    assert is_unitary(U)

    # Step 1 in [2].
    U_mb = Phi_dag @ U @ Phi
    Psi, eig_values = orthonormal_eigensystem(U_mb.T @ U_mb)
    eps = 0.5 * np.angle(eig_values)

    # Step 3 in [2].
    Psi_tilde = U_mb @ Psi @ np.diag(np.exp(-1j * eps))
    assert is_real(Psi_tilde)

    # Go back from magical to computational basis.
    Psi = Phi @ Psi
    Psi_tilde = Phi @ Psi_tilde
    assert is_maximally_entangled_basis(Psi)
    assert is_maximally_entangled_basis(Psi_tilde)

    # Step 2 in [2].
    VA, VB, xi = decompose_4x4_partial(Psi)

    # Step 4 in [2].
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

    result.append(GateSingle(Gate2('Rz', 0.5 * np.pi), 1))
    result.append(GateFC(Gate2('X'), 0))
    result.append(GateSingle(Gate2('Rz', t1), 0))
    result.append(GateSingle(Gate2('Ry', t2), 1))
    result.append(GateFC(Gate2('X'), 1))
    result.append(GateSingle(Gate2('Ry', t3), 1))
    result.append(GateFC(Gate2('X'), 0))
    result.append(GateSingle(Gate2('Rz', -0.5 * np.pi), 0))

    N = magic_N(a)
    assert np.allclose(N, gates_to_matrix(result, 2) * np.exp(0.25j * np.pi))

    return result


def decompose_4x4_optimal(U):
    """Builds optimal decomposition of general 4x4 unitary matrix.

    This decomposition consists of at most 3 CNOT gates, 15 Rx/Ry gates and one
    R1 gate.

    Returns list of `Gate`s.
    """
    assert is_unitary(U)

    magic_decomp = decompose_to_magic_diagonal(U)

    result = []
    result += apply_on_qubit(su_to_gates(magic_decomp['VA']), 1)
    result += apply_on_qubit(su_to_gates(magic_decomp['VB']), 0)
    result += decompose_magic_N(magic_decomp['alpha'])
    result += apply_on_qubit(su_to_gates(magic_decomp['UA']), 1)
    result += apply_on_qubit(su_to_gates(magic_decomp['UB']), 0)

    # Adding global phase using Rz and R1.
    gl_phase = magic_decomp['global_phase'] + 0.25 * np.pi
    if np.abs(gl_phase) > 1e-9:
        result.append(GateSingle(Gate2('Rz', 2 * gl_phase), 0))
        result.append(GateSingle(Gate2('R1', 2 * gl_phase), 0))

    result = skip_identities(result)

    assert _allclose(U, gates_to_matrix(result, 2))

    return result
