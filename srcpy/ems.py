import numpy as np
from qutip import expect, liouvillian, vector_to_operator, qeye, Qobj

from utils import DATA_PATH, TOL
from model import build_system


__all__ = ['metastable_states']


def metastable_states(g2, eta, D, nl_eta, nl_dis, dim, g1=1, adag=False):
    r"""Computes the metast states of the oscillator.

    Parameters
    ----------
    g2 : float
        the nonlinear dissipative rate.
    eta : float
        the driving strength.
    D : float
        the detuning.
    nl_eta : int
        the power of the driving.
    nl_dis : int
        the power of the dissipation.
    dim : int
        the cut-off dimension of the Hilbert space.
    g1 : float
        the linear dissipative rate (the default is 1).
    adag : bool
        wether to use $a$ or $a^\dagger$ in the linear lindblad term (the default is False).

    Returns
    -------
    ems
        list of metastable states, where the first one corresponds to the steady state (size `nl_eta + 1`).
    left
        list of left eigenstates after normalisation (size `nl_eta`).
    right
        list of right eigenstates after normalisation (size `nl_eta`).
    projectors
        list of left projectors onto the metastable states, in the same order as `ems` (size `nl_eta`).
    """
    dirpath = DATA_PATH / 'ems' / f'{"adag" if adag else "a"}/{nl_eta}_{nl_dis}'
    fname = f'{g1}_{g2:.6f}_{eta:.6f}_{D}_{dim}'

    files = ['EMS', 'L', 'R', 'P']
    if all((dirpath / f'{fname}-{post}.npy').exists() for post in files):
        EMS, Ls, Rs, Ps = [np.load(dirpath / f'{fname}-{post}.npy') for post in files]
        return list(map(Qobj, EMS)), list(map(Qobj, Ls)), list(map(Qobj, Rs)), list(map(Qobj, Ps))
    else:
        L = build_system(g1, g2, eta, D, nl_eta=nl_eta, nl_dis=nl_dis, dim=dim, adag=adag)
        output = ems_qvdp(L, nl_eta=nl_eta, max_eigvals=nl_eta + 2)
        dirpath.mkdir(parents=True, exist_ok=True)
        for j, post in enumerate(files):
            np.save(dirpath / f'{fname}-{post}.npy', np.array(output[j]))
        return output


def ems_qvdp(L_or_H, Js=None, nl_eta=2, max_eigvals=5):
    L = liouvillian(L_or_H, Js) if Js else L_or_H
    eigvals, eigvecsr = L.eigenstates(sort='high', eigvals=max_eigvals)
    _, eigvecsl = L.dag().eigenstates(sort='high', eigvals=max_eigvals)

    rho_ss = vector_to_operator(eigvecsr[0])
    rho_ss = rho_ss / rho_ss.tr()

    Id = qeye(rho_ss.shape[0])

    if nl_eta == 2:
        return ems_qvdp_2(eigvals, eigvecsl, eigvecsr, rho_ss, Id)
    elif nl_eta == 3:
        return ems_qvdp_3(eigvals, eigvecsl, eigvecsr, rho_ss, Id)
    elif nl_eta == 4:
        return ems_qvdp_4(eigvals, eigvecsl, eigvecsr, rho_ss, Id)
    else:
        raise NotImplementedError


def ems_qvdp_2(eigv, evl, evr, rho_ss, Id):
    l1 = vector_to_operator(evl[1]).tidyup(atol=TOL)
    r1 = vector_to_operator(evr[1]).tidyup(atol=TOL)

    l1 = (l1 + l1.dag()) * 0.5
    r1 = (r1 + r1.dag()) * 0.5

    l1 /= expect(l1, r1)
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
    l1 = vector_to_operator(evl[1]).tidyup(atol=TOL)
    r1 = vector_to_operator(evr[1]).tidyup(atol=TOL)
    l2 = vector_to_operator(evl[2]).tidyup(atol=TOL)
    r2 = vector_to_operator(evr[2]).tidyup(atol=TOL)

    sgn = -1 if np.imag(ev[1]) > 0 else 1
    L, R = [(l1 + l2) / 2, sgn * 1j * (l1 - l2) / 2], [(r1 + r2) / 2, sgn * 1j * (r1 - r2) / 2]
    ems_ev = []
    for k in range(2):
        L[k] /= expect(L[k], R[k])

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

    return mus, [l1], [r1], projs


def ems_qvdp_4(ev, evl, evr, rho_ss, Id):
    L, R = [], []
    ems_ev = []
    Dcs = []
    sgn = -1 if np.imag(ev[1]) > 0 else 1

    l1 = vector_to_operator(evl[1]).tidyup(atol=TOL)
    l2 = vector_to_operator(evl[2]).tidyup(atol=TOL)

    L.append(0.5 * (l1 + l2))
    L.append(0.5 * sgn * 1j * (l1 - l2))
    L.append(vector_to_operator(evl[3]).tidyup(atol=TOL))

    r1 = vector_to_operator(evr[1]).tidyup(atol=TOL)
    r2 = vector_to_operator(evr[2]).tidyup(atol=TOL)

    R.append(0.5 * (r1 + r2))
    R.append(0.5 * sgn * 1j * (r1 - r2))
    R.append(vector_to_operator(evr[3]).tidyup(atol=TOL))
    for j in range(3):
        L[j] /= expect(L[j], R[j])
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
