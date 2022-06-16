import numpy as np
from qutip import destroy, expect, liouvillian, vector_to_operator

from utils import TOL, DATA_PATH


def build_system(g1, g2, eta, D, nl_eta=2, nl_dis=2, dim=25, full_lv=True, phi=0, adag=False):
    """Construct the liouvillian operator for the system.

    Parameters
    ----------
    g1 : float
        linear dissipation rate
    g2 : float
        non-linear dissipation rate, corresponding to `nl_dis`
    eta : float
        driving strength
    D : float
        detuning frequency between oscillator and driving
    nl_eta : int
         power of the driving (the default is 2).
    nl_dis : int
         power of the non-linear dissipation (the default is 2).
    dim : int
         boson truncsation level (the default is 25).
    full_lv : bool
         whether to return the Liouvillian superoperator or the hamiltonian and jump operators separately
         (the default is True).
    phi : float
         driving phase (the default is 0).
    adag : bool (deprecated)
         wether to use creation or anihilation operator for the linear jump operator (the default is False).

    Returns
    -------
    L or (H, J)
        Liouvillian superoperator or the hamiltonian and jump operators (see `full_lv`).
    """
    a = destroy(dim)
    phase = np.exp(1j * phi * nl_eta)
    H = D * a.dag() * a + 1j * eta * (a**nl_eta * phase - a.dag()**nl_eta * np.conj(phase))
    Js = []
    if g1 > 0:
        Js.append(np.sqrt(g1) * (a.dag() if adag else a))
    if g2 > 0:
        Js.append(np.sqrt(g2) * a**nl_dis)
    return liouvillian(H, Js) if full_lv else (H, Js)


def eigvals(g2, eta, D, nl_eta, nl_dis, dim=50, n_eigvals=8, g1=1):
    """Computes the eigenvalues of the system.

    Parameters
    ----------
    n_eigvals : int
        number of eigenvalues to calculate

    Returns
    -------
    ev: list of complex numbers
        the eigenvalues sorted in decreasing real part and decreasing imaginary part.
    """
    saved_ev = max(n_eigvals, 10)
    fpath = DATA_PATH / 'ev/a' / f'{nl_eta}_{nl_dis}' / f'{g1}_{g2}_{eta}_{D}_{dim}_{saved_ev}.npy'

    if fpath.exists():
        return np.load(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    L = build_system(1, g2, eta, D, nl_eta, nl_dis, dim)
    ev = L.eigenenergies(sort='high', eigvals=saved_ev)
    ev[0] = 0

    # sort ev againg such that the positive imaginary part always goes before the negative
    ev = sorted(ev, key=lambda x: (-np.real(x), np.imag(x)))
    np.save(fpath, ev)
    return ev[:n_eigvals]


def _diag_liouv(L_or_H, Js=None, nl_eta=2, max_eigvals=5):
    L = liouvillian(L_or_H, Js) if Js else L_or_H
    eigvals, eigvecsr = L.eigenstates(sort='high', eigvals=max_eigvals)
    _, eigvecsl = L.dag().eigenstates(sort='high', eigvals=max_eigvals)

    rho_ss = vector_to_operator(eigvecsr[0])
    rho_ss = rho_ss / rho_ss.tr()

    # normalise eigenvectors
    rv = []
    lv = []

    j = 1
    while j < max_eigvals:
        if np.abs(np.imag(eigvals[j])) < TOL:  # real
            l1 = vector_to_operator(eigvecsl[j]).tidyup(atol=TOL)
            r1 = vector_to_operator(eigvecsr[j]).tidyup(atol=TOL)

            l1 = (l1 + l1.dag()) * 0.5
            r1 = (r1 + r1.dag()) * 0.5

            l1 /= expect(l1, r1)

            rv.append(r1)
            lv.append(l1)
            j += 1
        else:
            l1 = vector_to_operator(eigvecsl[j]).tidyup(atol=TOL)
            r1 = vector_to_operator(eigvecsr[j]).tidyup(atol=TOL)
            if j + 1 >= max_eigvals:
                l2 = l1.dag()
                r2 = r1.dag()
            else:
                l2 = vector_to_operator(eigvecsl[j + 1]).tidyup(atol=TOL)
                r2 = vector_to_operator(eigvecsr[j + 1]).tidyup(atol=TOL)

            sgn = -1 if np.imag(eigvals[j]) > 0 else 1

            lr = (l1 + l2) * 0.5
            w = np.exp(-1j * 2 * np.pi / nl_eta)
            # cambiant + <-> - obtens un lobe o laltre
            li = (l1 * np.conj(w) + l2 * w) * 0.5 * sgn
            rr = (r1 + r2) * 0.5
            ri = (r1 * np.conj(w) + r2 * w) * 0.5 * sgn

            lr /= expect(lr, rr)
            li /= expect(li, ri)

            rv.append(rr)
            lv.append(lr)
            rv.append(ri)
            lv.append(li)

            j += 2

    extreme_ev = []
    for j in range(max_eigvals - 1):
        print(j, rv[j].check_herm(), lv[j].check_herm())
        evlv = lv[j].eigenenergies()
        extreme_ev.append((evlv[0], evlv[-1]))  # (min, max)
    print(extreme_ev)

    return rho_ss, eigvals, lv, rv, extreme_ev


def classicallity(left_ops, ems):
    """Performs the classicallity test.

    Parameters
    ----------
    left_ops : list of QObj
        the left eigenstates of the Liouvillian
    ems : list of QObj
        the metastable states

    Returns
    -------
    C: np.ndarray
        coefficient matrix where $[C]_{ij} = \tr(L_i ρ_j)
    cl: float
        classicallity correction
    projs: list of QObj
        the projectors onto the metastable phases
    """
    n = len(ems)

    if len(left_ops) != n:
        raise ValueError(f'Number of metastable states ({n}) is different from'
                         f'number of left eigenvectors ({len(left_ops)}).')

    C = np.zeros((n, n), dtype=complex)

    for j in range(n):
        for k in range(n):
            C[j, k] = expect(left_ops[j], ems[k])

    Cinv = np.linalg.inv(C)
    projs_dual = [sum(Cinv[j, k] * left_ops[k] for k in range(n)) for j in range(n)]
    pl_min = [Pl.eigenenergies()[0] for Pl in projs_dual]
    C_cl = 2 * sum(-pl for pl in pl_min)
    projs = [(Pl - pl * left_ops[0]) / (1 + C_cl * 0.5) for Pl, pl in zip(projs_dual, pl_min)]

    return C, np.real(C_cl), projs


def evolution_matrix(left_ops, ems):
    """Computes the classical evolution matrix in the metastable transient.

    Parameters
    ----------
    left_ops : list of QObj
        the left eigenvectors
    ems : list of QObj
        the metastable phases

    Returns
    -------
    np.ndarray
        the matrix

    """
    C, *_ = classicallity(left_ops, ems)
    n, _ = C.shape

    Cinv = np.linalg.inv(C)
    Mdual = np.array([
        [sum(Cinv[j, n] * eigvals[n] * C[n, k] for n in range(n)) for k in range(n)]
        for j in range(n)
    ])
    M = np.maximum(Mdual, 0)
    for j in range(n):
        M[j, j] = Mdual[j, j] + sum(min(Mdual[k, j], 0) for k in range(n) if k != j)

    return M
