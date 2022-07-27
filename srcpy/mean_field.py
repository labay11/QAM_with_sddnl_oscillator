import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm as Normalize
from scipy.optimize import root

from utils import PLOT_PATH, amplitude, latexify

latexify(plt, type='beamer43', fract=(0.45, 0.4))


def mf(R, phi, g1, g2, eta, D, n, m):
    dR = -0.5 * g1 * R - 0.5 * m * g2 * R**(2 * m - 1) - n * eta * R**(n - 1) * np.cos(n * phi)
    dP = -D + n * eta * R**(n - 2) * np.sin(n * phi)
    return dR, dP


def _mf(x, g1, g2, eta, D, n, m):
    return [*mf(x[0], x[1], g1, g2, eta, D, n, m)]


def plot(fig, ax, g2, eta, D, n, m, g1=1, colorbar=False):
    mu = amplitude(g2, eta, n, m)
    Rs = np.linspace(0, 2 * mu, 100)
    phis = np.linspace(0, 2 * np.pi, 100)
    Rs, phis = np.meshgrid(Rs, phis)

    dR, dP = mf(Rs, phis, g1, g2, eta, D, n, m)
    res = root(_mf, [mu, 1 * np.pi / n], args=(g1, g2, eta, D, n, m))

    U = np.abs(dR)
    norm = Normalize(max(1e-6, np.amin(U)), vmin=np.amin(U), vmax=np.amax(U))
    cb = ax.streamplot(Rs, phis, dR, dP, color=U, norm=norm, cmap='Greens', density=1.2, arrowsize=0.5)
    if colorbar:
        fig.colorbar(cb.lines)

    ax.scatter([mu] * n, [(2 * j + 1) * np.pi / n for j in range(n)], c='k', marker='o', label='Mean field')
    ax.scatter([res.x[0]] * n, [(2 * j + 1) * np.pi / n for j in range(n)], c='r', marker='x', label='Numerical')
    ax.set(yticks=np.arange(0, 5, 1) * np.pi / 2, yticklabels=[
        r'$0$' if j == 0 else
        rf'${(j if j > 1 else "") if j % 2 == 1 else (j//2 if j > 2 else "")}\pi{"/2" if j % 2 == 1 else ""}$'
        for j in np.arange(0, 5, 1)])

    ax.set(xlabel=r'$R$',
           ylabel=r'$\phi$',
           yticks=np.pi * np.arange(0, 2.5, 0.5))

    ax.text(0.96, 0.945, f'$n = {n}$\n$m = {m}$', bbox=dict(facecolor='white', alpha=0.8, boxstyle='Round'),
            transform=ax.transAxes, va='top', ha='right')


def single_plot(params):
    latexify(plt, type='beamer43', fract=(0.8, 0.7))

    fig, ax = plt.subplots()
    plot(fig, ax, **params)
    ax.legend(loc='upper left')
    fig.savefig(PLOT_PATH / 'mean_field' / 'mf_single.pdf')


def full_plot(params):
    latexify(plt, type='beamer43', fract=(1., 0.7))
    nrows, ncols = np.array(params).shape

    fig, axs = plt.subplots(nrows, ncols, sharey=True, gridspec_kw={'wspace': 0.05})

    for j in range(nrows):
        for k in range(ncols):
            plot(fig, axs[j, k], **params[j][k])
            if j < nrows - 1:
                axs[j, k].set_xlabel('')
            if k > 0:
                axs[j, k].set_ylabel('')

    fig.savefig(PLOT_PATH / 'mean_field' / 'mf_full.pdf')


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
    # plot(**p42)
    #
    #
    # for p in [p22, p32, p33, p43, p44]:
    #     plot(**p)
    single_plot(p33)
    # full_plot([[p22, p32, p33], [p42, p43, p44]])
