import numpy as np
import matplotlib.pyplot as plt

from utils import amplitude, latexify, PLOT_PATH


def comparison(exponents, g2=0.4):
    latexify(plt, type='paper', fract=0.25, palette='default_mk')

    fig, ax = plt.subplots()
    ax.set(xlabel=r'$\eta_n / \gamma_m$', ylabel='Mean field amplitude')

    x = np.linspace(0, 2, 500)

    for nl_eta, nl_dis in exponents:
        y = amplitude(g2, x * g2, nl_eta, nl_dis)
        ax.plot(x, y, label=f'${nl_eta} - {nl_dis}$', markevery=249)

    ax.legend(title=r'$n - m$', ncol=2)
    fig.savefig(PLOT_PATH / 'amplitude_comparison.pdf')


if __name__ == '__main__':
    comparison([(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (4, 5)])
