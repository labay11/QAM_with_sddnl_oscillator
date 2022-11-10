"""Utility function to plot the Fock occupation probability of a state."""
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.stats import poisson

from model import build_system
from utils import PLOT_PATH, latexify, amplitude


def plot_fock(rho, fig=None, ax=None, figsize=(6, 6), draw_dist=True, cut_off=None, labels=True):
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if qt.isket(rho):
        rho = qt.ket2dm(rho)

    diag = np.real(rho.diag())
    n_levels = cut_off or len(diag)

    N = np.arange(0, len(diag), 1) - 0.5

    ax.bar(N[:n_levels], diag[:n_levels], color='C0')
    mu = None
    if draw_dist:
        mu = np.sqrt(np.sum(diag * (N + 0.5)))
        ax.plot(N[:n_levels], poisson.pmf(N[:n_levels] + 0.5, mu**2), c='C1', ls='-', label=rf'$\beta = {mu:.4f}$')

    if labels:
        ax.set_xlabel('Fock level')
        ax.set_ylabel('Level probability')

    return N, diag, mu


def plot_multiple_fock(fig, axs, states, **kwargs):
    res = [
        plot_fock(state, fig, ax, **kwargs)
        for ax, state in zip(axs, states)
    ]
    return res


if __name__ == '__main__':
    p22 = {
        'g2': 0.3, 'D': 0.4, 'n': 2, 'm': 2, 'eta': 0.3 * 3.73
    }
    p32 = {
        'g2': 0.4, 'D': 0.4, 'n': 3, 'm': 2, 'eta': 0.5
    }
    p33 = {
        'g2': 0.6, 'D': 0.4, 'n': 3, 'm': 3, 'eta': 2.
    }
    p42 = {
        'g2': 0.4, 'D': 0.4, 'n': 4, 'm': 2, 'eta': 0.4 * 3.4
    }
    p43 = {
        'g2': 0.2, 'D': 0.4, 'n': 4, 'm': 3, 'eta': 0.4 * 3.4
    }
    p44 = {
        'g2': 0.1, 'D': 0.4, 'n': 4, 'm': 4, 'eta': 1.1423118881998557
    }

    params = [[p22, p32, p33], [p42, p43, p44]]

    latexify(plt, type='beamer43', fract=(1., 0.7), palette='qualitative')
    nrows, ncols = np.array(params).shape

    fig, axs = plt.subplots(nrows, ncols, sharey='row', gridspec_kw={'wspace': 0.05})

    for j in range(nrows):
        for k in range(ncols):
            params[j][k]['dim'] = 50
            params[j][k]['g1'] = 1
            n = params[j][k]['n']
            m = params[j][k]['m']
            L = build_system(**params[j][k])

            N, diag = plot_fock(qt.steadystate(L), fig, axs[j, k], color='C0')

            beta = amplitude(params[j][k]['g2'], params[j][k]['eta'], n, m)
            if n < 2 * m:
                axs[j, k].plot(N, poisson.pmf(N + 0.5, beta**2), c='C2', ls='--', label=rf'$R = {beta:.4f}$')
            axs[j, k].text(0.96, 0.4, f'$n = {n}$\n$m = {m}$',
                           bbox=dict(facecolor='white', edgecolor='None', alpha=0.8, boxstyle='Round'),
                           transform=axs[j, k].transAxes, va='center', ha='right')
            axs[j, k].legend()

            if j < nrows - 1:
                axs[j, k].set_xlabel('')
            if k > 0:
                axs[j, k].set_ylabel('')

    fig.savefig(PLOT_PATH / 'fock' / 'fock_full.pdf')
