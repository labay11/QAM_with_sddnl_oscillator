import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import qutip as qt


def wigner_ss(state, xvec):
    W0 = qt.wigner(state, xvec, xvec, method='clenshaw')
    W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
    return W


def plot_wigner(rho, fig=None, ax=None, figsize=(6, 6),
                cmap=None, alpha_max=7.5, colorbar=False,
                method='clenshaw', title=None, norm=None):
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if qt.isket(rho):
        rho = qt.ket2dm(rho)

    xvec = np.linspace(-alpha_max, alpha_max, 200)
    W0 = qt.wigner(rho, xvec, xvec, method=method)

    W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)

    # wlim = abs(W).max()
    if norm is None:
        wmin, wmax = W.min(), W.max()
    else:
        wmin, wmax = norm

    if cmap is None:
        cmap = 'magma'
        ax.tick_params(axis='both', which='both', color='w')

    # cf = ax.contourf(xvec, yvec, W, 100, norm=mpl.colors.Normalize(wmin, wmax), cmap=cmap)
    cf = ax.pcolormesh(xvec, yvec, W,
                       cmap='magma', norm=mpl.colors.Normalize(wmin, wmax), shading='gouraud', rasterized=True)

    if xvec is not yvec:
        ax.set_ylim(xvec.min(), xvec.max())

    ax.set_xlabel(r'$\rm{Re}(\alpha)$')
    ax.set_ylabel(r'$\rm{Im}(\alpha)$')

    if colorbar:
        fmt = ScalarFormatter()
        fmt.set_powerlimits((0, 0))
        fig.colorbar(cf, ax=ax, format=fmt)

    ax.set_title(title if title else "Wigner function")

    return fig, ax, cf


def plot_multiple_wigner(fig, axs, states, alpha_max=7., div=500, colorbar=True, cmap='magma'):
    if isinstance(alpha_max, float):
        alpha_max = (-alpha_max, alpha_max)
    if isinstance(alpha_max, tuple):
        alpha_max = [alpha_max] * len(states)
    if isinstance(alpha_max, list):
        alpha_max = [a if isinstance(a, tuple) else (-a, a) for a in alpha_max]

    wigners = []
    xvecs = []
    wmin, wmax = np.inf, -np.inf
    for state, alphas in zip(states, alpha_max):
        xvec = np.linspace(*alphas, div)
        W = wigner_ss(state, xvec)
        wigners.append(W)
        xvecs.append(xvec)
        w1, w2 = W.min(), W.max()
        wmin = min(wmin, w1)
        wmax = max(wmax, w2)

    norm = Normalize(wmin, wmax)
    for ax, W, xvec in zip(axs, wigners, xvecs):
        if not colorbar:
            norm = Normalize(W.min(), W.max())
        cf = ax.pcolormesh(xvec, xvec, W,
                           norm=norm, cmap=cmap, shading='gouraud', rasterized=True)
        ax.set_xlabel(r'$\rm{Re}(\alpha)$')
        ax.set_ylabel(r'$\rm{Im}(\alpha)$')
        ax.tick_params(axis='both', which='both', color='w')

    cax = None
    if colorbar:
        fmt = ScalarFormatter()
        fmt.set_powerlimits((0, 0))
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(cf, cax=cax, format=fmt)

    return cf, cax
