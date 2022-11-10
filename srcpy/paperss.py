import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from wigner import plot_multiple_wigner
from fock import plot_multiple_fock
from utils import latexify, local_data_path, local_plot_path, build_filename, amplitude
from constants import SQUEEZING


def fig_q():
    latexify(plt, type='preprint', fract=0.25)

    def ls(n, m):
        if m == n - 1:
            return ':'
        elif m == n + 1:
            return '--'
        else:
            return '-'

    fig, axs = plt.subplots(ncols=3, nrows=2, sharex='col', gridspec_kw={'wspace': 0.15, 'hspace': 0.05})

    dirpath = local_data_path('squeezing_ss')
    mnms = sorted(
        [tuple(map(int, folder.name.split('_'))) for folder in dirpath.iterdir()],
        key=lambda x: (x[0], x[1])
    )

    X = np.linspace(0.0001, 6, 100)
    fname = build_filename(g2=0.2, D=0.4, betas=X) + '.npy'

    for n_ in [2, 3, 4]:
        fpaths = [(dirpath / f'{n}_{m}' / fname, m) for n, m in mnms if n == n_]

        j = 0
        for file, m in fpaths:
            if not file.exists():
                continue

            params = SQUEEZING[(n_, m)]
            print(n_, m, params)
            ev = np.load(file)
            sq = (ev[:, 1] - ev[:, 0]**2) / ev[:, 0] - 1
            axs[0, n_ - 2].plot(X**2, np.real(ev[:, 0]), label=str(m), ls=ls(n_, m), c=f'C{j}')
            axs[1, n_ - 2].plot(X**2, np.real(sq), label=str(m), ls=ls(n_, m), c=f'C{j}')

            R = amplitude(params['g2'], params['eta'], n_, m) if not (n_ == 2 and m == 1) else 2.4
            idx = np.argmin(np.abs(R - X))
            axs[0, n_ - 2].scatter(R**2, np.real(ev[idx, 0]), marker='o', c='grey')
            axs[1, n_ - 2].scatter(R**2, np.real(sq[idx]), marker='o', c='grey')
            j += 1

        axs[0, n_ - 2].text(0.5, 0.98, rf'$({chr(97 + n_ - 2)})\ n = {n_}$',
                            va='top', ha='center', transform=axs[0, n_ - 2].transAxes)
        axs[1, n_ - 2].text(0.5, 0.98, rf'$({chr(97 + 3 + n_ - 2)})$',
                            va='top', ha='center', transform=axs[1, n_ - 2].transAxes)
        axs[0, n_ - 2].plot(X**2, X**2, c='k')
        axs[1, n_ - 2].axhline(0.0, c='k')
        axs[1, n_ - 2].set_xlabel(r'$R^2$')
        axs[0, n_ - 2].legend(title=r'$m$')

    axs[1, 0].set_ylabel(r'$\mathcal{Q}$')
    axs[0, 0].set_ylabel(r'$\langle\hat{n}\rangle$')

    fig.savefig(local_plot_path(__file__) / 'q_mandel_ss.pdf')


def fig_w():
    latexify(plt, type='preprint', fract=0.5)

    fig, axs = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    ss = []
    for (n, m), params in SQUEEZING.items():
        print(n, m, params)
        L = build_system(**params, dim=100, n=n, m=m)
        ss.append(qt.steadystate(L))

    axs_squeezed = axs.reshape(-1)
    plot_multiple_wigner(fig, axs_squeezed, ss, colorbar=False, labels=False)
    for j in range(3):
        axs[2, j].set_xlabel(r'$\rm{Re}(\alpha)$')
        axs[j, 0].set_ylabel(r'$\rm{Im}(\alpha)$')

    for (n, m), ax in zip(SQUEEZING.keys(), axs_squeezed):
        ax.text(0.99, 0.98, rf'$n = {n}$' '\n' rf'$m = {m}$', c='white',
                va='top', ha='right', transform=axs[n - 2, n - m + 1].transAxes)

    fig.savefig(local_plot_path(__file__) / 'wigner_ss.pdf')


def fig_distr():
    latexify(plt, type='preprint', fract=0.4)

    fig, axs = plt.subplots(ncols=3, nrows=3, sharex=True, sharey='row', gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    ss = []
    for (n, m), params in SQUEEZING.items():
        print(n, m, params)
        L = build_system(**params, dim=100, n=n, m=m)
        ss.append(qt.steadystate(L))
        axs[n - 2, n - m + 1].text(0.99, 0.98, rf'$n = {n}$' '\n' rf'$m = {m}$',
                                   va='top', ha='right', transform=axs[n - 2, n - m + 1].transAxes)

    axs_squeezed = axs.reshape(-1)
    plot_multiple_fock(fig, axs_squeezed, ss, labels=False, cut_off=40)
    for j in range(3):
        axs[2, j].set_xlabel(r'$n$')
        axs[j, 0].set_ylabel(r'$p_n$')

    fig.savefig(local_plot_path(__file__) / 'fock_ss.pdf')


fig_distr()
fig_w()
