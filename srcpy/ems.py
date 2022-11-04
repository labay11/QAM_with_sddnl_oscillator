import numpy as np
import qutip as qt

from utils import TOL, local_data_path, is_zero
from model import build_system


__all__ = ['metastable_states']


CLEAN_TOL = 1e-12


def metastable_states(g2, eta, D, n, m, dim=50, g1=1):
    r"""Computes the metast states of the oscillator.

    Parameters
    ----------
    g2 : float
        the nonlinear dissipative rate.
    eta : float
        the driving strength.
    D : float
        the detuning.
    n : int
        the power of the driving.
    m : int
        the power of the dissipation.
    dim : int
        the cut-off dimension of the Hilbert space.
    g1 : float
        the linear dissipative rate (the default is 1).

    Returns
    -------
    ems
        list of metastable states, where the first one corresponds to the steady state (size `n + 1`).
    left
        list of left eigenstates after normalisation (size `n`).
    right
        list of right eigenstates after normalisation (size `n`).
    projectors
        list of left projectors onto the metastable states, in the same order as `ems` (size `n`).

    Raises
    ------
    ValueError
        if it is not possible to claculate the metastable states because the shape of the eigenvalue distribution
        is not valid for the oscillator parameters. Example: for n = m = 2` the first eigenvalue is not real.
    """
    dirpath = local_data_path(__file__, n, m)
    fname = f'{g1}_{g2}_{eta}_{D}_{dim}'

    files = ['EMS', 'L', 'R', 'P']
    if all((dirpath / f'{fname}-{post}.npy').exists() for post in files):
        EMS, Ls, Rs, Ps = [np.load(dirpath / f'{fname}-{post}.npy') for post in files]
        return list(map(qt.Qobj, EMS)), list(map(qt.Qobj, Ls)), list(map(qt.Qobj, Rs)), list(map(qt.Qobj, Ps))
    else:
        L = build_system(g1, g2, eta, D, n=n, m=m, dim=dim)
        output = ems_qvdp(L, n=n, max_eigvals=n + 2)
        for j, post in enumerate(files):
            np.save(dirpath / f'{fname}-{post}.npy', np.array(output[j]))
        return output


def ems_qvdp(L_or_H, Js=None, n=2, max_eigvals=5):
    L = qt.liouvillian(L_or_H, Js) if Js else L_or_H
    eigvals, eigvecsr = L.eigenstates(sort='high', eigvals=max_eigvals)
    _, eigvecsl = L.dag().eigenstates(sort='high', eigvals=max_eigvals)

    rho_ss = qt.vector_to_operator(eigvecsr[0])
    rho_ss = rho_ss / rho_ss.tr()

    Id = qt.qeye(rho_ss.shape[0])

    eigopsl = [qt.vector_to_operator(evl).tidyup(atol=CLEAN_TOL) for evl in eigvecsl]
    eigopsr = [qt.vector_to_operator(evr).tidyup(atol=CLEAN_TOL) for evr in eigvecsr]

    if n == 2:
        return ems_qvdp_2(eigvals, eigopsl, eigopsr, rho_ss, Id)
    elif n == 3:
        return ems_qvdp_3(eigvals, eigopsl, eigopsr, rho_ss, Id)
    elif n == 4:
        return ems_qvdp_4(eigvals, eigopsl, eigopsr, rho_ss, Id)
    else:
        raise NotImplementedError


def ems_qvdp_2(eigv, evl, evr, rho_ss, Id):
    if not is_zero(np.imag(eigv[1])):
        raise ValueError(f'First eigenvalue must be real: {eigv[1]}')

    l1 = (evl[1] + evl[1].dag()) * 0.5
    r1 = (evr[1] + evr[1].dag()) * 0.5

    l1 /= qt.expect(l1, r1)
    eigv1 = l1.eigenenergies()
    c1min, c1max = eigv1[0], eigv1[-1]

    mus = [
        rho_ss + c1min * r1,
        rho_ss + c1max * r1
    ]

    Dc1 = c1max - c1min
    projs = [
        (-l1 + c1max * Id) / Dc1,
        (l1 - c1min * Id) / Dc1
    ]

    return mus, [l1], [r1], projs


def ems_qvdp_3(ev, evl, evr, rho_ss, Id):
    if is_zero(np.imag(ev[1])) or is_zero(np.imag(ev[2])):
        raise ValueError(f'First and second eigenvalue must be complex: {ev[1]}, {ev[2]}')
    if np.abs(ev[1] - np.conj(ev[2])) > TOL:
        raise ValueError(f'First and second are not conjugate of each other: {ev[1]}, {ev[2]}')

    sgn = -1 if np.imag(ev[1]) > 0 else 1
    L = [(evl[1] + evl[2]) / 2, sgn * 1j * (evl[1] - evl[2]) / 2]
    R = [(evr[1] + evr[2]) / 2, sgn * 1j * (evr[1] - evr[2]) / 2]

    ems_ev = []
    for k in range(2):
        L[k] /= qt.expect(L[k], R[k])

        evlv = L[k].eigenenergies()
        ems_ev.append((evlv[0], evlv[-1]))  # (min, max)

    mus = [
        rho_ss + ems_ev[0][0] * R[0],
        rho_ss + ems_ev[0][1] * R[0] + ems_ev[1][1] * R[1],
        rho_ss + ems_ev[0][1] * R[0] + ems_ev[1][0] * R[1]
    ]

    Dc1 = ems_ev[0][1] - ems_ev[0][0]
    Dc2 = ems_ev[1][1] - ems_ev[1][0]
    projs = [
        (-L[0] + ems_ev[0][1] * Id) / Dc1,
        (L[1] - (ems_ev[1][0] / Dc1) * (L[0] - ems_ev[0][0] * Id)) / Dc2,
        (-L[1] + (ems_ev[1][1] / Dc1) * (L[0] - ems_ev[0][0] * Id)) / Dc2
    ]

    return mus, L, R, projs


def ems_qvdp_4(ev, evl, evr, rho_ss, Id):
    if is_zero(np.imag(ev[1])) or is_zero(np.imag(ev[2])):
        raise ValueError(f'First and second eigenvalue must be complex: {ev[1]}, {ev[2]}')
    if not is_zero(np.imag(ev[3])):
        raise ValueError(f'Third eigenvalue must be real: {ev[3]}')
    if np.abs(ev[1] - np.conj(ev[2])) > TOL:
        raise ValueError(f'First and second are not conjugate of each other: {ev[1]}, {ev[2]}')

    ems_ev = []
    Dcs = []
    sgn = -1 if np.imag(ev[1]) > 0 else 1

    L = [0.5 * (evl[1] + evl[2]), 0.5 * sgn * 1j * (evl[1] - evl[2]), evl[3]]
    R = [0.5 * (evr[1] + evr[2]), 0.5 * sgn * 1j * (evr[1] - evr[2]), evr[3]]

    for j in range(3):
        L[j] /= qt.expect(L[j], R[j])
        evlv = L[j].eigenenergies()
        ems_ev.append((evlv[0], evlv[-1]))  # (min, max)
        Dcs.append(evlv[-1] - evlv[0])

    mus = [
        rho_ss + ems_ev[0][0] * R[0] + ems_ev[2][1] * R[2] + ems_ev[1][0] * R[1],
        rho_ss + ems_ev[0][1] * R[0] + ems_ev[2][0] * R[2] + ems_ev[1][0] * R[1],
        rho_ss + ems_ev[0][1] * R[0] + ems_ev[2][1] * R[2] + ems_ev[1][1] * R[1],
        rho_ss + ems_ev[0][0] * R[0] + ems_ev[2][0] * R[2] + ems_ev[1][1] * R[1]
    ]

    projs = [
        (-L[0]/Dcs[0] - L[1]/Dcs[1] + L[2]/Dcs[2] +
         Id * (ems_ev[0][1]/Dcs[0] + ems_ev[1][0]/Dcs[1] - ems_ev[2][0] / Dcs[2])) * 0.5,
        (L[0]/Dcs[0] - L[1]/Dcs[1] - L[2]/Dcs[2] +
         Id * (-ems_ev[0][0]/Dcs[0] + ems_ev[1][1]/Dcs[1] + ems_ev[2][0] / Dcs[2])) * 0.5,
        (L[0]/Dcs[0] + L[1]/Dcs[1] + L[2]/Dcs[2] -
         Id * (ems_ev[0][0]/Dcs[0] + ems_ev[1][1]/Dcs[1] + ems_ev[2][0] / Dcs[2])) * 0.5,
        (-L[0]/Dcs[0] + L[1]/Dcs[1] - L[2]/Dcs[2] +
         Id * (ems_ev[0][0]/Dcs[0] - ems_ev[1][0]/Dcs[1] + ems_ev[2][1] / Dcs[2])) * 0.5
    ]

    return mus, L, R, projs


if __name__ == '__main__':
    # from constants import POINTS, DELTA
    #
    # for n, points in POINTS.items():
    #     for g2, eta, _ in points:
    #         for dim in [10, 15, 20, 30, 40, 50]:
    #             try:
    #                 metastable_states(g2, eta, DELTA, n, n, dim=dim)
    #             except:
    #                 print(n, g2, eta, dim)
    from wigner import plot_multiple_wigner
    from utils import driving_dissipation_ratio
    import matplotlib.pyplot as plt

    betas = np.linspace(0.0001, 10, 100)[::10]
    n = 3
    m = 2
    g2 = 0.2
    D = 0.4

    for beta in betas:
        eta = driving_dissipation_ratio(beta, n, m) * g2
        dim = min(max(int(round(beta**2)), 40), 100)
        ems, *_ = metastable_states(g2, eta, D, n, m, dim)
        num = qt.num(dim)
        num2 = num**2
        print([(em.tr(), qt.expect(num, em), qt.expect(num2, em)) for em in ems], eta, beta)
        fig, axs = plt.subplots(ncols=len(ems))
        plot_multiple_wigner(fig, axs, ems)
        plt.title(str(beta))
        plt.show()
