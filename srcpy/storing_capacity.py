import numpy as np
# from scipy.stats import poisson
from scipy.optimize import root
from scipy.special import loggamma
import matplotlib.pyplot as plt
from utils import latexify, local_plot_path

TOL = 1e-9


def k_max(k, alpha, tol):
    # return k * (np.log(alpha / k) - 1) - alpha + 9 * np.log(10)
    return - alpha**2 + 2 * k * np.log(alpha) - loggamma(k + 1) + tol * np.log(10)


def k_max_jac(k, alpha):
    return np.log(alpha / k) - 2


def storing_capacity(p, alphas, tol=1e-9, infidelity=True):
    alphas = np.linspace(0.5, 6, 1000)
    # M = np.arange(2, 80, 1)

    if not isinstance(p, list):
        p = [p]

    max_level = []
    for a in alphas:
        # x = poisson.pmf(M, a**2)
        # max_level.append(np.argmax((x - TOL) < 0))
        sol = root(k_max, 2 * a**2, args=(a, -np.log10(tol)))
        max_level.append(sol.x[0])

    max_level = np.array(max_level)

    return [_p * ((1 - np.exp(-alphas**2 * (1 - np.cos(2 * np.pi / _p)))) if infidelity else 1) / max_level
            for _p in p]


def simple_plot(ps):
    latexify(plt, type='beamer43', fract=(0.8, 0.6), palette='qualitative')

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r'$\beta$')
    # ax.set_ylabel(r'$\alpha_c$')
    ax.set_ylabel('Storage capacity')
    alphas = np.linspace(0.5, 6, 1000)
    capacities = storing_capacity(ps, alphas, TOL)
    capacities_2 = storing_capacity(ps, alphas, TOL, infidelity=False)

    c = 0
    for p, sc, sc2 in zip(ps, capacities, capacities_2):
        # ax.plot(alphas, sc, label=f'{p}', c=f'C{c}')
        ax.plot(alphas, sc2, ls='--', c=f'C{c}', label=f'{p}')
        c += 1

    ax.axhline(0.138, c='k')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='upper right', borderaxespad=0.1, title=r'$n$')
    ax.text(alphas[-1], 0.14, 'Hebbian', va='bottom', ha='right')
    # ax.text(0.995, 0.98, r'$(b)$', ha='right', va='top', transform=ax.transAxes)
    # fig.tight_layout(pad=0.05)
    fig.patch.set_alpha(0)
    fig.savefig(local_plot_path(__file__) / 'critical_unbounded.pdf')


def plot_como(ps):
    latexify(plt, paper=2, fract=0.18, margin_x=0, margin_y=0, palette='qualitative')

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r'$\beta$')
    # ax.set_ylabel(r'$\alpha_c$')
    ax.set_ylabel('Storage capacity')
    alphas = np.linspace(0.5, 6, 1000)
    capacities = storing_capacity(ps, alphas, TOL)

    c = 0
    for p, sc in zip(ps, capacities):
        ax.plot(alphas, sc, label=f'{p}', c=f'C{c}')
        c += 1

    ax.axhline(0.138, c='k')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(0.99, 0.98),
              loc='upper right', borderaxespad=0, title='Number of patterns', ncol=2)
    fig.tight_layout(pad=0.05)
    fig.patch.set_alpha(0)
    fig.savefig('critical_como.pdf')


if __name__ == '__main__':
    simple_plot([2, 3, 4, 5])
