import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from wigner import plot_multiple_wigner
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


comparison_g1()
