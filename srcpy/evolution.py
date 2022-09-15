import numpy as np
import matplotlib.pyplot as plt

from model import eigvals
from utils import latexify, local_data_path, local_plot_path
from constants import POINTS


def plot_time(ax, a_func, n, g2, eta, dim, D=0.4, color=None, inline_label=None):
    ev = eigvals(g2, eta, D, n, n, dim)

    fpath = local_data_path(__file__, n, n) / f'g-{g2}_e-{eta}_d-{D}_t-0-4999.999999999999-50000.npy'

    if not fpath.exists():
        return

    data = np.load(fpath)
    ax.plot(np.real(data[:, 0]), a_func(data[:, 1]), c=color, label=inline_label)
    # ax.set_xlim(left=0.01)

    t0 = -1 / np.real(ev[n])
    t1 = -1 / np.real(ev[n - 1])
    ax.axvspan(t0, t1, color=color, alpha=0.2, ec=None)

    # if n == 4:
    #     ymin, ymax = min(0.5 * j, 0.75), 0.5 + 0.25 * j
    #     ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, color=c, alpha=0.2, ec=None)
    # else:
    #     ymin, ymax = j / 3, (j + 1) / 3
    #     ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, color=c, alpha=0.2, ec=None)

    # if inline_label:
    #     ax.text(0.2 + 0.2 * j,
    #             ymin * 0.85 + ymax * 0.15 if j == 0 else ymin * 0.95 + ymax * 0.05,
    #             rf'$\tau_{n} = {inline_label}$',
    #             c=color, ha='center', va='bottom', transform=ax.transAxes)


def plot_evolution(ns, a_func=np.abs):
    latexify(plt, type='paper', fract=0.15 * len(ns), palette='qualitative')

    fig, axs = plt.subplots(nrows=len(ns), sharex=True)

    k = 0
    for ax, n in zip(axs, ns):
        j = 0
        for g2, eta, dim in POINTS[n]:
            plot_time(ax, a_func, n, g2, eta, dim, color=f'C{j}', inline_label=rf'$\tau_{n} = {int(10**j):d}$')
            j += 1

        ax.text(1, 0.99, rf'$({chr(97 + k)})\ n = {n}$', ha='right', va='top', transform=ax.transAxes)
        ax.set(
            xlabel=r'$\gamma_1 t$' if k > 0 else '',
            ylabel=r'$|\langle a \rangle|$',
            xscale='log')

        k += 1

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0))

    fig.tight_layout()
    fig.savefig(local_plot_path(__file__) / f'me_{"_".join(map(str, ns))}.pdf')


plot_evolution([3, 4])
