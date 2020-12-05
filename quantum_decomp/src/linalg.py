import numpy as np

from quantum_decomp.src.utils import is_orthogonal, is_real


def _dot(a, b):
    return np.dot(a.conj(), b)


def clear_global_phase(x):
    angle = 0.0
    for val in x:
        if np.abs(val) > 1e-9:
            angle = np.angle(val)
    x = x * np.exp(-1j * angle)
    return x


def project_on_orth_subspace(v1, v2):
    """Projects vector v1 on subspace, orthogonal to vector v2."""
    v1_pr_v2 = (_dot(v2, v1) / _dot(v2, v2)) * v2
    ans = v1 - v1_pr_v2
    return ans / np.linalg.norm(ans)


def find_real_basis(vecs):
    """Finds real orthonormal basis in a linear subspace."""
    ans = []
    while len(vecs) > 0:
        v2 = -1
        for i in range(len(vecs)):
            v_real = clear_global_phase(vecs[i])
            if is_real(v_real):
                ans.append(v_real)
                v2 = i
                break
        assert v2 != -1
        vecs = [
            project_on_orth_subspace(
                vecs[i],
                vecs[v2]) for i in range(
                len(vecs)) if i != v2]
    return ans


def orthonormal_eigensystem(A):
    """Builds orthogonal eigensystem for symmetric matrix A.

    Returns Q and d, such that Q is orthogonal and A = Q * diag(d) * Q^T.
    """
    n = A.shape[0]
    assert np.allclose(A.T, A)

    eig_values, eig_vecs = np.linalg.eig(A)

    eig_vals_to_vecs = {}
    for i in range(n):
        ev = eig_values[i]
        exists = False
        for key in eig_vals_to_vecs.keys():
            if np.allclose(key, ev):
                exists = True
                ev = key
                break
        if not exists:
            eig_vals_to_vecs[ev] = [eig_vecs[:, i]]
        else:
            eig_vals_to_vecs[ev].append(eig_vecs[:, i])

    Q = []
    d = []
    for eig_val, eig_space in eig_vals_to_vecs.items():
        d += [eig_val] * len(eig_space)
        Q += find_real_basis(eig_space)

    d = np.array(d)
    Q = np.real(np.array(Q).T)

    assert is_orthogonal(Q)
    assert np.allclose(A, Q @ np.diag(d) @ Q.T)

    return Q, d
