import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qutip import destroy, rand_ket, coherent, expect, mcsolve
from qutip.solver import Options
from qutip.fileio import qsave, qload
from joblib import Parallel, delayed

from model import build_system
from utils import local_data_path, local_plot_path, amplitude
from wigner import wigner_ss


def _rand_ket(m, a=None, amp_max=None):
    if amp_max is None:
        return rand_ket(m)

    j = 0
    while j < 100:
        r = rand_ket(m)
        if np.abs(expect(a, r)) <= amp_max:
            return r
        j += 1

    raise RuntimeError('No state could be found with matching conditions.')


def evolve(times, phi0, g1, g2, eta, D, n, m, dim):
    H, J = build_system(g1, g2, eta, D, n, m, dim, full_lv=False)

    opts = Options(num_cpus=4,
                   store_states=True,
                   store_final_state=True)
    data = mcsolve(H, phi0, times, c_ops=J, ntraj=1, options=opts)

    fname = local_data_path(__file__, n, m) / f'g1-{g1}_g2-{g2}_eta-{eta}_d-{D}_t-{times[-1]}-{len(times)}'
    qsave(data, str(fname))

    return data, fname


def animate_trajectory(fname, alpha_max=7.5):
    data = qload(str(fname))

    states = data.states[0]
    times = data.times
    N = len(states)

    fig, ax = plt.subplots(figsize=(6, 5))

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    xvec = np.linspace(-alpha_max, alpha_max, 200)

    def _calc_wigner(i):
        return wigner_ss(states[i], xvec)

    _res = Parallel(n_jobs=5, verbose=1)(delayed(_calc_wigner)(i) for i in range(N))
    Rss = np.array(_res)
    norm = Normalize(Rss.min(), Rss.max())

    cf = ax.pcolormesh(xvec, xvec, Rss[0],
                       norm=norm, cmap='magma', shading='nearest', rasterized=True)
    fig.colorbar(cf, cax=cax)
    ax.set_xlabel(r'$\rm{Re}(\alpha)$')
    ax.set_ylabel(r'$\rm{Im}(\alpha)$')
    ax.set_title(r'$t = 0$')

    def animate(i):
        ax.clear()
        ax.pcolormesh(xvec, xvec, Rss[i],
                      norm=norm, cmap='magma', shading='nearest', rasterized=True)
        ax.set_title(rf'$t = {times[i]}$')

        return []

    interval = 0.5  # in seconds
    ani = animation.FuncAnimation(fig, animate, N, interval=interval * 1e3, blit=True)

    n, m = map(int, fname.parent.name.split('_'))
    ani.save(local_plot_path(__file__, n, m) / (fname.name[:-3] + '.mp4'))

    return ani


if __name__ == '__main__':
    params = {
        'g1': 0,
        'g2': 0.02,
        'eta': 3.,
        'D': 0.4,
        'n': 4,
        'm': 4,
        'dim': 50
    }

    beta = amplitude(**params)

    n = params['n']

    theta_mid = 2 * np.pi / n
    phi0 = coherent(params['dim'], beta * np.exp(1j * theta_mid))

    times = np.linspace(0, 1e3, 10000)

    data, fpath = evolve(times, phi0, **params)
    animate_trajectory(fpath)
