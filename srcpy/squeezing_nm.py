from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.stats import poisson


from model import build_system
from utils import local_data_path, local_plot_path, latexify, amplitude, driving_dissipation_ratio
from wigner import plot_wigner
from fock import plot_fock


def plot_comparison(params, titles=None, cut_off=None):
    latexify(plt, type='preprint', fract=0.4)

    fig, axs = plt.subplots(nrows=2, ncols=len(params), sharey='row')

    for j, param in enumerate(params):
        L = build_system(**param)
        rho = qt.steadystate(L)

        plot_wigner(rho, fig=fig, ax=axs[0, j])
        N, _ = plot_fock(rho, fig=fig, ax=axs[1, j], cut_off=cut_off)
        R = amplitude(**param)
        axs[1, j].plot(N[:cut_off], poisson.pmf(N[:cut_off] + 0.5, R**2), c='C2', label=rf'$R = {R:.4f}$')
        axs[1, j].legend()

        if j > 0:
            axs[0, j].set_ylabel('')
            axs[1, j].set_ylabel('')

        axs[1, j].set_title('')
        if titles:
            axs[0, j].set_title(titles[j])

    fig.savefig(local_plot_path(__file__) / 'squeezing_wf.pdf')


if __name__ == '__main__':
    params_3 = [
        {'g1': 1, 'D': 0.4, 'g2': 1, 'eta': 1, 'n': 3, 'm': 2, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.4, 'eta': 3, 'n': 3, 'm': 3, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.1, 'eta': 2, 'n': 3, 'm': 4, 'dim': 100},
    ]

    params_3_above = [
        {'g1': 1, 'D': 0.4, 'g2': 0.1, 'eta': 2, 'n': 3, 'm': 4, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.02, 'eta': driving_dissipation_ratio(2.5, 3, 5)*0.02, 'n': 3, 'm': 5, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.005, 'eta': driving_dissipation_ratio(2.5, 3, 6)*0.005, 'n': 3, 'm': 6, 'dim': 100},
    ]

    params_4 = [
        {'g1': 1, 'D': 0.4, 'g2': 0.1, 'eta': 0.5, 'n': 4, 'm': 3, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.01, 'eta': 1.5, 'n': 4, 'm': 4, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.05, 'eta': 5, 'n': 4, 'm': 5, 'dim': 100},
    ]

    params_5 = [
        {'g1': 1, 'D': 0.4, 'g2': 0.4, 'eta': 0.5, 'n': 5, 'm': 3, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.01, 'eta': 0.2, 'n': 5, 'm': 4, 'dim': 100},
        {'g1': 1, 'D': 0.4, 'g2': 0.05, 'eta': 5, 'n': 5, 'm': 5, 'dim': 100},
    ]

    plot_comparison(params_3_above, [rf'$n = {p["n"]}\ \& \ m = {p["m"]}$' for p in params_3_above], cut_off=50)
