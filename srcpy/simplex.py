from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import qutip as qt
from joblib import Parallel, delayed

from model import build_system, classicallity
from ems import metastable_states
from utils import local_data_path, local_plot_path, latexify


DATA_PATH = local_data_path(__file__)
PLOT_PATH = local_plot_path(__file__)


def calc_coefs(ops, state):
    return [qt.expect(op, state) for op in ops]


def draw_simplex_2d(fig, ax, rho_ss, ems, Ls, points):
    c_ss = calc_coefs(Ls, rho_ss)
    c_ems = np.array([calc_coefs(Ls, e) for e in ems])

    ax.scatter(points[:, 0], points[:, 1], marker='.', c='C0')

    ax.scatter(c_ss[0], c_ss[1], marker='x', c='C1', label=r'$\rho_{ss}$')
    ax.add_patch(Polygon(c_ems, edgecolor='C2', fill=False))
    ax.scatter(c_ems[:, 0], c_ems[:, 1], marker='o', c='C2', label='EMS')

    ax.set_xlabel(r'$c_1$')
    ax.set_ylabel(r'$c_2$')
    ax.legend()

    _, C, _ = classicallity(Ls, ems)
    ax.text(0.02, 0.98, r'$\mathcal{C}_{cl} = ' + f'{C:.2g}$', va='top', ha='left', transform=ax.transAxes)

    return fig, ax


def draw_simplex_1d(fig, ax, rho_ss, ems, Ls, points):
    c_ss = np.real(calc_coefs(Ls, rho_ss))
    c_ems = np.array([np.real(calc_coefs(Ls, e)) for e in ems])

    # path = Path(np.real(c_ems), closed=True)
    # points_inside = path.contains_points(np.real(points))
    # colors = ['b' if inside else 'k' for inside in points_inside]
    n_points = len(points)
    ax.scatter(points[:, 0], np.random.rand(n_points), marker='.', c='C0')

    ax.axvline(c_ss[0], c='C1', label=r'$\rho_{ss}$', ls='--')
    for cm in c_ems:
        ax.axvline(cm[0], c='C2')

    ax.plot([], [], ls='-', c='C2', label='EMS')
    ax.set_xlabel(r'$c_1$')
    ax.set_yticks([])
    ax.legend()

    _, C, _ = classicallity(Ls, ems)
    ax.text(0.02, 0.98, r'$\mathcal{C}_{cl} = ' + f'{C:.2g}$', va='top', ha='left', transform=ax.transAxes)

    return fig, ax


def draw_simplex_3d(fig, ax, rho_ss, ems, Ls, points):
    ax.remove()
    ax = fig.add_subplot(projection='3d')

    c_ss = np.real(calc_coefs(Ls, rho_ss))
    c_ems = np.real(np.array([calc_coefs(Ls, e) for e in ems]))

    # path = Path(np.real(c_ems), closed=True)
    # points_inside = path.contains_points(np.real(points))
    # colors = ['b' if inside else 'k' for inside in points_inside]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.', c='C0', depthshade=True)

    ax.scatter(c_ss[0], c_ss[1], c_ss[2], c='C1', marker='x', label=r'$\rho_{ss}$', depthshade=False)
    # ax.add_collection3d(Poly3DCollection(c_ems, alpha=0.2, facecolors='g', edgecolors='g'))
    ax.plot_trisurf(c_ems[:, 0], c_ems[:, 1], c_ems[:, 2], color='C2', alpha=0.3)

    ax.scatter(c_ems[:, 0], c_ems[:, 1], c_ems[:, 2], c='C2', marker='o', label='EMS', depthshade=True)

    ax.set_xlabel(r'$c_1$')
    ax.set_ylabel(r'$c_2$')
    ax.set_zlabel(r'$c_3$')

    return fig, ax


def plot_simplex(g2, eta, D, n, m, dim, g1=1, n_points=5000, append=False):
    latexify(plt, type='beamer43', fract=(0.48, 0.4), palette='qualitative')

    ems, Ls, Rs, projs = metastable_states(g2, eta, D, n, m, dim)
    rho_ss = qt.steadystate(build_system(g1, g2, eta, D, n, m, dim))

    fpath = local_data_path(__file__, n, m) / f'{g2}_{eta}_{D}_{dim}.npy'
    points = None
    if not fpath.exists() or append:
        _res = Parallel(n_jobs=10, verbose=1)(delayed(calc_coefs)(Ls, qt.rand_ket(dim)) for _ in range(n_points))
        points = np.array(_res)

    if not fpath.exists():
        np.save(fpath, points)
    elif points is not None:
        old_points = np.load(fpath)
        points = np.concatenate((old_points, points), axis=0)
        np.save(fpath, points)
    else:
        points = np.load(fpath)

    points = np.real(points)

    fig, ax = plt.subplots()

    if n == 2:
        draw_simplex_1d(fig, ax, rho_ss, ems, Ls, points)
    elif n == 3:
        draw_simplex_2d(fig, ax, rho_ss, ems, Ls, points)
    elif n == 4:
        draw_simplex_3d(fig, ax, rho_ss, ems, Ls, points)

    fig.savefig(local_plot_path(__file__, n, m) / f'{g2}_{eta}_{D}_{dim}.pdf')


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

    params = [p22, p32, p33, p43, p44]

    for param in params:
        plot_simplex(**param, dim=50)
