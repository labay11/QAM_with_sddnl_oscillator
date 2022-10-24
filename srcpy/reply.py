import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from wigner import plot_multiple_wigner
from thermo_limit import plot_thermodynamic_limit
from utils import latexify, local_plot_path
from constants import POINTS


def comparison_g1(g1s=[0, 1e-8, 1e-4, 1e-2], n=4, point=1):
    latexify(plt, type='preprint', fract=0.2)

    fig, axs = plt.subplots(ncols=len(g1s), sharey=True, gridspec_kw={'wspace': 0.02})

    params = {
        'g2': POINTS[n][point][0],
        'eta': POINTS[n][point][1],
        'D': 0.4,
        'n': n,
        'm': n,
        'dim': 50
    }

    states = []

    for g1 in g1s:
        params['g1'] = g1
        L = build_system(**params)
        states.append(qt.steadystate(L))

    plot_multiple_wigner(fig, axs, states)

    for ax, g1 in zip(axs, g1s):
        label = '0' if g1 == 0 else ('10^{' f'{int(np.log10(g1))}' '}')
        ax.text(0.5, 0.99, rf'$\gamma_1 = {label}$', ha='center', va='top', color='w', transform=ax.transAxes)
        ax.set_ylabel('')

    axs[0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.savefig(local_plot_path(__file__) / 'comparison_g1.pdf')


def comparison_thermodynamic():
    latexify(plt, type='preprint', fract=0.3)

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, gridspec_kw={'wspace': 0.12, 'hspace': 0.05})

    plot_thermodynamic_limit(np.linspace(0, 0.5, 100), 0.4, 3, 3, 200, fig=fig, axs=axs[:, 0])
    plot_thermodynamic_limit(np.linspace(0, 0.5, 100), 0.4, 4, 4, 200, fig=fig, axs=axs[:, 1])

    for j in range(2):
        axs[j, 1].set_ylabel('')
        axs[1, j].text(0.1, 0.8, rf'$n = {j + 3}$', va='center', ha='left', transform=axs[1, j].transAxes)

    fig.savefig(local_plot_path(__file__) / 'comparison_thermodynamic.pdf')


comparison_thermodynamic()
