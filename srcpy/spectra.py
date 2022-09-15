from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from model import eigvals
from utils import latexify, MARKERS, local_plot_path, local_data_path
from constants import POINTS


def spectra_comparison(upper_ev=5, only_mm=False):
    latexify(plt, type='beamer43', fract=(0.6, 0.55), palette='qualitative')

    fig, axs = plt.subplots(ncols=2 if not only_mm else 1, nrows=2, sharex=only_mm)

    lbl = [2, 10, 100]

    for ax, n in zip(axs.T, [3, 4]):
        if only_mm:
            ax.set_ylabel(r'Im $\lambda$')
            ax.set_xscale('symlog', linthresh=1e-4)
            ax.set_yscale('symlog', linthresh=1e-4)
            ax.text(0.99, 0.95, rf'$({chr(97 + n - 3)})\ n = {n}$', transform=ax.transAxes, va='top', ha='right')
        else:
            ax[1].set_xlabel(r'Re $\lambda$')
            ax[0].set_xscale('symlog', linthresh=1e-4)
            ax[1].set_xscale('symlog', linthresh=1e-4)
            ax[1].set_yscale('symlog', linthresh=1e-4)
            ax[0].text(0.99, 0.95, rf'$({chr(97 + n - 3)})\ n = {n}$', transform=ax[0].transAxes, va='top', ha='right')

        for j, (g2, eta, _) in enumerate(POINTS[n]):
            ev = eigvals(g2, eta, 0.4, n, n, dim=50, n_eigvals=n + upper_ev)
            if only_mm:
                ax.scatter(np.real(ev[:n]), np.imag(ev[:n]),
                           c=f'C{j}', marker=MARKERS[j], label=rf'$\tau_n = {lbl[j]}$')
            else:
                ax[0].scatter(np.real(ev[:n]), np.imag(ev[:n]),
                              c=f'C{j}', marker=MARKERS[j], label=rf'$\tau_n = {lbl[j]}$')
                ax[0].scatter(np.real(ev[n:]), np.imag(ev[n:]),
                              edgecolors=f'C{j}', marker=MARKERS[j], facecolors='none')
                ax[1].scatter(np.real(ev[:n]), np.imag(ev[:n]), c=f'C{j}', marker=MARKERS[j])

    if only_mm:
        axs[1].set_xlabel(r'Re $\lambda$')
        handles, labels = axs[0].get_legend_handles_labels()
    else:
        axs[0, 0].set_ylabel(r'Im $\lambda$')
        axs[1, 0].set_ylabel(r'Im $\lambda$')

        handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, borderaxespad=0.02,
               ncol=len(labels), loc='upper center')

    plt.tight_layout(pad=0.02, rect=[0, 0, 1, 0.88])
    plt.savefig(local_plot_path(__file__) / 'beamer_spectra.pdf')


def spectra_evolution(param, values, n_ev=8, fig=None, ax=None, **kwargs):
    if fig is None or ax is None:
        latexify(plt, type='paper', fract=0.4, palette='sequential')
        fig, ax = plt.subplots()
        ax.set_xlabel(r'Re $\lambda$')
        ax.set_ylabel(r'Im $\lambda$')
        ax.set_xscale('symlog', linthresh=1e-1)

    nv = len(values) - 1
    cmap = cm.get_cmap('viridis')

    for j in range(len(values)):
        print(j)
        kwargs[param] = values[j]
        ev = eigvals(**kwargs, n_eigvals=n_ev)
        print(ev)
        n = kwargs['n']
        print(np.log10(np.real(ev[n - 1])/np.real(ev[n])))
        ax.scatter(np.real(ev[:n]), np.imag(ev[:n]), c=[cmap(j / nv)], marker=MARKERS[j], label=rf'${values[j]}$')
        ax.scatter(np.real(ev[n:]), np.imag(ev[n:]), edgecolors=[cmap(j / nv)], marker=MARKERS[j], facecolors='none')

    fig.legend(title=r'$\eta$')
    plt.tight_layout(pad=0.02)

    dirpath = local_plot_path(__file__, kwargs["n"], kwargs["m"])
    plt.savefig(dirpath / f'{kwargs["g2"]}_{kwargs["eta"]}_{kwargs["D"]}.pdf')


def prsentation_plot():
    # 2x2 plot with 3-2, 3-3, 4-3, 4-4
    latexify(plt, type='beamer43', fract=1., palette='sequential')
    fig, axs = plt.subplots(2, 2)
    for j in range(2):
        axs[j, 0].set_ylabel(r'Im $\lambda$')
        axs[0, j].set_xlabel(r'Re $\lambda$')


p43 = {
    'g2': 0.2, 'D': 0.4, 'n': 4, 'm': 3, 'eta': 0.4 * 3.4, 'dim': 50, 'g1': 1
}
p32 = {
    'g2': 0.4, 'D': 0.4, 'n': 3, 'm': 2, 'eta': 1, 'dim': 80
}
etas = [0, 1e-4, 1e-2, 1e-1]

# spectra_evolution('eta', etas, n_ev=5, **p43)
spectra_comparison(only_mm=True)
