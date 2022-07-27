import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from utils import PLOT_PATH, latexify, driving_dissipation_ratio
from wigner import plot_wigner, plot_multiple_wigner


def single_oscillator():
    latexify(plt, type='beamer43', fract=(0.35, 0.4))

    a = qt.destroy(100)
    D = 0.4
    H = D * a.dag() * a

    ev, eket = H.groundstate()

    fig, ax = plt.subplots()

    plot_wigner(eket, fig, ax, alpha_max=4)
    ax.set_title('')

    fig.savefig(PLOT_PATH / 'beamer' / 'single_oscillator.pdf')


def single_driven_oscillator():
    # latexify(plt, type='beamer43', fract=(0.35, 0.4))

    a = qt.destroy(100)
    D = 0.4

    for eta in np.linspace(0.01, 1, 10):
        # eta = 0.1
        H = D * a.dag() * a + 1j * eta * (a**2 - a.dag()**2)

        ev, eket = H.groundstate()

        fig, ax = plt.subplots()

        plot_wigner(eket, fig, ax, alpha_max=4)
        plt.show()
    # ax.set_title('')
    #
    # fig.savefig(PLOT_PATH / 'beamer' / 'single_driven_oscillator.pdf')


def single_driven_dissipative_oscillator():
    latexify(plt, type='beamer43', fract=(0.35, 0.4))

    a = qt.destroy(100)
    D = 0.4
    eta = 0.1
    H = D * a.dag() * a + 1j * eta * (a**2 - a.dag()**2)
    J = [a]
    L = qt.liouvillian(H, J)

    ss = qt.steadystate(L)

    fig, ax = plt.subplots()

    plot_wigner(ss, fig, ax, alpha_max=4)
    ax.set_title('')

    fig.savefig(PLOT_PATH / 'beamer' / 'single_driven_dissipative_oscillator.pdf')


def single_driven_amplificative_oscillator():
    latexify(plt, type='beamer43', fract=(0.35, 0.4))

    a = qt.destroy(100)
    D = 0.4
    eta = driving_dissipation_ratio(2.0, 2, 2) * 0.5
    H = D * a.dag() * a + 1j * eta * (a**2 - a.dag()**2)
    J = [a.dag(), 0.5 * a**2]
    L = qt.liouvillian(H, J)

    ss = qt.steadystate(L)

    fig, ax = plt.subplots()

    plot_wigner(ss, fig, ax, alpha_max=6)
    ax.set_title('')

    fig.savefig(PLOT_PATH / 'beamer' / 'single_driven_amplificative_oscillator.pdf')


def single_driven_dissipative_nonlinear_oscillator():
    latexify(plt, type='beamer43', fract=(0.35, 0.4))

    a = qt.destroy(100)
    D = 0.4
    eta = driving_dissipation_ratio(2.0, 2, 2) * 0.5
    H = D * a.dag() * a + 1j * eta * (a**2 - a.dag()**2)
    J = [a, 0.5 * a**2]
    L = qt.liouvillian(H, J)

    ss = qt.steadystate(L)

    fig, ax = plt.subplots()

    plot_wigner(ss, fig, ax, alpha_max=6)
    ax.set_title('')

    fig.savefig(PLOT_PATH / 'beamer' / 'single_driven_dissipative_nonlinear_oscillator.pdf')


def single_driven_nonlinear_oscillator():
    latexify(plt, type='beamer43', fract=(0.5, 0.4))

    a = qt.destroy(100)
    D = 0.4
    eta = driving_dissipation_ratio(2.0, 2, 2) * 0.5
    H = D * a.dag() * a + 1j * eta * (a**2 - a.dag()**2)
    J = [0.5 * a**2]
    L = qt.liouvillian(H, J)

    ss = qt.steadystate(L)

    fig, ax = plt.subplots()

    plot_wigner(ss, fig, ax, alpha_max=6, colorbar=True)
    ax.set_title('')

    fig.savefig(PLOT_PATH / 'beamer' / 'single_driven_nonlinear_oscillator.pdf')

    latexify(plt, type='beamer43', fract=(1., 0.4))

    a = qt.destroy(100)
    D = 0.4

    ss = []
    params = [(3, 3, 0.4), (3, 2, 1), (4, 4, 0.1)]
    for n, m, g2 in params:
        eta = driving_dissipation_ratio(2.5, n, m) * g2
        H = D * a.dag() * a + 1j * eta * (a**n - a.dag()**n)
        J = [g2 * a**m]
        L = qt.liouvillian(H, J)

        ss.append(qt.steadystate(L))

    fig, axs = plt.subplots(ncols=3, sharey=True, gridspec_kw={'wspace': 0.01})

    plot_multiple_wigner(fig, axs, ss, alpha_max=7.5, colorbar=True, div=250)
    for ax, (n, m, g) in zip(axs, params):
        ax.set_title('')
        ax.set_ylabel('')
        ax.text(0.5, 0.99, rf'$n = {n}\ \&\ m = {m}$', va='top', ha='center', transform=ax.transAxes, color='w')
    axs[0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.savefig(PLOT_PATH / 'beamer' / 'single_driven_nonlinear_oscillator_full.pdf')


def betas(n, m, betas, g2=0.1):
    latexify(plt, type='beamer43', fract=(1., 0.4))

    a = qt.destroy(100)
    D = 0.4

    ss = []
    for beta in betas:
        eta = driving_dissipation_ratio(beta, n, m) * g2
        H = D * a.dag() * a + 1j * eta * (a**n - a.dag()**n)
        J = [a, g2 * a**m]
        L = qt.liouvillian(H, J)

        ss.append(qt.steadystate(L))

    fig, axs = plt.subplots(ncols=3, sharey=True, gridspec_kw={'wspace': 0.01})

    plot_multiple_wigner(fig, axs, ss, alpha_max=7.5, colorbar=False, div=400)
    for ax, beta in zip(axs, betas):
        ax.set_title('')
        ax.set_ylabel('')
        ax.text(0.5, 0.99, rf'$\beta = {beta}$', va='top', ha='center', transform=ax.transAxes, color='w')
    axs[0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.savefig(PLOT_PATH / 'beamer' / f'betas_{n}_{m}.pdf')


def full():
    latexify(plt, type='beamer43', fract=(1., 0.4))

    a = qt.destroy(100)
    D = 0.4

    ss = []
    params = [(3, 3, 0.4), (3, 2, 1), (4, 4, 0.1)]
    for n, m, g2 in params:
        eta = driving_dissipation_ratio(2.5, n, m) * g2
        H = D * a.dag() * a + 1j * eta * (a**n - a.dag()**n)
        J = [a, g2 * a**m]
        L = qt.liouvillian(H, J)

        ss.append(qt.steadystate(L))

    fig, axs = plt.subplots(ncols=3, sharey=True, gridspec_kw={'wspace': 0.01})

    plot_multiple_wigner(fig, axs, ss, alpha_max=7.5, colorbar=False, div=250)
    for ax, (n, m, g) in zip(axs, params):
        ax.set_title('')
        ax.set_ylabel('')
        ax.text(0.5, 0.99, rf'$n = {n}\ \&\ m = {m}$', va='top', ha='center', transform=ax.transAxes, color='w')
    axs[0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.savefig(PLOT_PATH / 'beamer' / 'full.pdf')


def full_equal():
    latexify(plt, type='beamer43', fract=(1., 0.4))

    a = qt.destroy(100)
    D = 0.4

    ss = []
    params = [(3, 3, 0.4), (4, 4, 0.1), (5, 5, 0.1)]
    for n, m, g2 in params:
        eta = driving_dissipation_ratio(2.5, n, m) * g2
        H = D * a.dag() * a + 1j * eta * (a**n - a.dag()**n)
        J = [a, g2 * a**m]
        L = qt.liouvillian(H, J)

        ss.append(qt.steadystate(L))

    fig, axs = plt.subplots(ncols=3, sharey=True, gridspec_kw={'wspace': 0.01})

    plot_multiple_wigner(fig, axs, ss, alpha_max=7.5, colorbar=False, div=250)
    for ax, (n, m, g) in zip(axs, params):
        ax.set_title('')
        ax.set_ylabel('')
        ax.text(0.5, 0.99, rf'$n = {n}\ \&\ m = {m}$', va='top', ha='center', transform=ax.transAxes, color='w')
    axs[0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.savefig(PLOT_PATH / 'beamer' / 'full_equal.pdf')


def plot_params(params):
    latexify(plt, type='beamer43', fract=(1., 0.7))

    nrows, ncols = np.array(params).shape

    ss = []
    for j in range(nrows):
        for k in range(ncols):
            params[j][k]['g1'] = 1
            params[j][k]['dim'] = 100
            L = build_system(**params[j][k])
            ss.append(qt.steadystate(L))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True,
                            gridspec_kw={'wspace': 0.01, 'hspace': 0.05})

    plot_multiple_wigner(fig, axs.reshape(-1), ss, alpha_max=7.5, colorbar=False, div=250)
    for j in range(nrows):
        for k in range(ncols):
            n = params[j][k]['n']
            m = params[j][k]['m']
            axs[j, k].text(0.5, 0.99, rf'$n = {n}\ \&\ m = {m}$',
                           va='top', ha='center', transform=axs[j, k].transAxes, color='w')
            if j < nrows - 1:
                axs[j, k].set_xlabel('')
            if k > 0:
                axs[j, k].set_ylabel('')

    fig.savefig(PLOT_PATH / 'beamer' / 'wigner_full.pdf')


# single_oscillator()
# single_driven_oscillator()
# single_driven_dissipative_oscillator()
# single_driven_amplificative_oscillator()
# single_driven_nonlinear_oscillator()
# single_driven_dissipative_nonlinear_oscillator()
# betas(4, 4, [0.5, 2, 4])
full()
full_equal()


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

# plot_params([[p22, p32, p33], [p42, p43, p44]])
