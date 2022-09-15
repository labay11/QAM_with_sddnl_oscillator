import numpy as np
# from scipy.stats import poisson
from scipy.optimize import root
from scipy.stats import linregress
from scipy.special import loggamma
import matplotlib.pyplot as plt
from utils import latexify, local_plot_path, MARKERS

TOL = 1e-9


def k_max(k, alpha, tol):
    # return k * (np.log(alpha / k) - 1) - alpha + 9 * np.log(10)
    return - alpha**2 + 2 * k * np.log(alpha) - loggamma(k + 1) + tol * np.log(10)


def k_max_jac(k, alpha):
    return np.log(alpha / k) - 2


def storing_capacity(p, alphas, tol=1e-9, infidelity=True):
    """Calculates the storage capacity of the quantum system.

    The storage capacity is defined as: $\alpha = patterns / dimension$.
    This definition has been scaled to also account for the distance between lobes,
    the scaled version is $\alpha' = \alpha [1 - F(\alpha)]$ where $F(\alpha)$ is
    the fidelity between two consecutives lobes.

    Parameters
    ----------
    p : int or list of ints
        the patterns to store
    alphas : list of floats
        the amplitudes of the lobes.
    tol : float
        the probability to consider a level as not occupied (the default is 1e-9).
    infidelity : bool
        wether to return the scaled fidelity or not (the default is True).

    Returns
    -------
    list of arrays
        for each `p`, returns the storage capacity for all `alphas`
    """
    if not isinstance(p, list):
        p = [p]

    max_level = []
    for a in alphas:
        sol = root(k_max, 2 * a**2, args=(a, -np.log10(tol)))
        max_level.append(sol.x[0])

    max_level = np.array(max_level)

    return [_p * ((1 - np.exp(-alphas**2 * (1 - np.cos(2 * np.pi / _p)))) if infidelity else 1) / max_level
            for _p in p]


def simple_plot(ps, fig=None, ax=None, show_top=False):
    savefig = False
    if fig is None or ax is None:
        latexify(plt, type='paper', fract=0.2, palette='default')
        fig, ax = plt.subplots(1, 1)
        savefig = True

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Storage capacity')
    alphas = np.linspace(0.5, 5, 1000)
    capacities = storing_capacity(ps, alphas, TOL)
    capacities_2 = storing_capacity(ps, alphas, TOL, infidelity=False)

    for p, sc, sc2 in zip(ps, capacities, capacities_2):
        c = p - 2
        ax.plot(alphas, sc, label=f'{p}', c=f'C{c}')
        ax.plot(alphas, sc2, ls='--', c=f'C{c}')

        if show_top:
            sc_max_j = np.argmax(sc)
            ax.scatter(alphas[sc_max_j], sc[sc_max_j], marker=MARKERS[c], c=f'C{c}')

    ax.axhline(0.138, c='k', ls=':')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='upper right', borderaxespad=0.1, title=r'$n$')
    ax.text(alphas[-1], 0.14, 'Hebbian', va='bottom', ha='right')
    # ax.text(0.995, 0.98, r'$(b)$', ha='right', va='top', transform=ax.transAxes)
    # fig.tight_layout(pad=0.05)
    # fig.patch.set_alpha(0)
    if savefig:
        fig.savefig(local_plot_path(__file__) / 'critical_unbounded.pdf')


def max_storage_capacity(ps, fig=None, ax=None, show_markers=True, show_generalized=False):
    savefig = False
    if fig is None or ax is None:
        latexify(plt, type='paper', fract=0.2, palette='default_mk')
        fig, ax = plt.subplots(1, 1)
        savefig = True

    ax.set_xlabel(r'\# of patterns')
    # ax.set_ylabel(r'$\alpha_c$')
    ax.set_ylabel('System size')
    alphas = np.linspace(1, 5, 500)
    capacities = [np.amax(sc) for sc in storing_capacity(ps, alphas, TOL)]

    ps = np.array(ps)
    dims = ps / capacities
    if show_markers:
        for p in ps:
            c = int(p - 2)
            ax.scatter(p, p / capacities[c], c=f'C{c}', marker=MARKERS[c])
    else:
        ax.plot(ps, dims, ls='-', label='Our')
    ax.plot(ps, ps / 0.138, ls=':', c='k', marker='None', label='Hebbian')

    res = linregress(dims, ps)
    print(res)

    if show_generalized:
        mind, maxd = ax.get_ylim()
        Ds = np.linspace(mind, maxd, 100)
        ax.plot(Ds / (2 * np.log(Ds)), Ds, ls='--', c='k', marker='None', label='Generalized')

    ax.legend()
    # ax.text(0.995, 0.98, r'$(b)$', ha='right', va='top', transform=ax.transAxes)
    # fig.tight_layout(pad=0.05)
    # fig.patch.set_alpha(0)
    if savefig:
        fig.savefig(local_plot_path(__file__) / 'critical_max_n.pdf')


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
    max_storage_capacity([2, 3, 4, 5, 6, 7, 8], show_markers=False, show_generalized=True)
    simple_plot([2, 3, 4, 5])
