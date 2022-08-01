import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import qutip as qt
from joblib import Parallel, delayed

from model import build_system
from wigner import wigner_ss
from utils import local_plot_path


def animate_ss(update_params, vals, const_params, alpha_max=7.5):
    N = len(vals[0])
    for val in vals:
        assert N == len(val)

    fig, ax = plt.subplots(figsize=(6, 5))

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    xvec = np.linspace(-alpha_max, alpha_max, 200)

    def _calc_eigvals(i):
        params = const_params.copy()
        for p, X in zip(update_params, vals):
            params[p] = X[i]
        L = build_system(**params)
        rhoss = qt.steadystate(L)
        return wigner_ss(rhoss, xvec)

    _res = Parallel(n_jobs=10, verbose=1)(delayed(_calc_eigvals)(i) for i in range(N))
    Rss = np.array(_res)
    norm = Normalize(Rss.min(), Rss.max())

    cf = ax.pcolormesh(xvec, xvec, Rss[0],
                       norm=norm, cmap='magma', shading='nearest', rasterized=True)
    fig.colorbar(cf, cax=cax)
    ax.set_xlabel(r'$\rm{Re}(\alpha)$')
    ax.set_ylabel(r'$\rm{Im}(\alpha)$')
    ax.set_title(', '.join(f'{p} = {X[0]:.4f}' for p, X in zip(update_params, vals)))

    def animate(i):
        ax.clear()
        ax.pcolormesh(xvec, xvec, Rss[i],
                      norm=norm, cmap='magma', shading='nearest', rasterized=True)
        ax.set_title(', '.join(f'{p} = {X[i]:.4f}' for p, X in zip(update_params, vals)))

        return []

    interval = 0.5  # in seconds
    ani = animation.FuncAnimation(fig, animate, N, interval=interval * 1e3, blit=True)
    return ani


if __name__ == '__main__':
    update_params = ['eta']
    const_params = {
        'g1': 0,
        'g2': 0.02,
        'eta': 3.,
        'D': 0.4,
        'n': 4,
        'm': 4,
        'dim': 50
    }
    ani = animate_ss(update_params, [np.linspace(0.01, 6, 50)], const_params)
    ani.save(local_plot_path(__file__, 3, 3) / ('_'.join(update_params) + '_'
             + '_'.join(f'{p}-{x}' for p, x in const_params.items() if p not in update_params) + '.mp4'))
