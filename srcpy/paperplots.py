import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import build_system, eigvals
from gap import plot_gap
from memory import plot_memory
from storing_capacity import simple_plot, max_storage_capacity
from utils import latexify, local_plot_path, local_data_path, MARKERS
from wigner import plot_multiple_wigner
from constants import POINTS


def fig1():
    latexify(plt, type='paper', fract=0.3)

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)

    data = [
        (0.2, 0.01, 3, 3), (1.5, 1.5 * 0.97, 3, 2),
        *[(0.2, 3**n * 0.2 / 2, n, n) for n in [3, 4]],
    ]

    print(data)

    states = []

    for g2, eta, n, m in data:
        L = build_system(1, g2, eta, 0.4, n, m, dim=150, adag=False)
        states.append(qt.steadystate(L))

    plot_multiple_wigner(fig, [axs[j, i] for i in range(2) for j in range(2)], states, colorbar=False)

    labels = [[r'$n = m = 3$'] * 2, [r'$n = 3\ \mathrm{and}\ m=2$', r'$n = m = 4$']]

    for j in range(2):
        for k in range(2):
            axs[j, k].set(title='', xlabel='', ylabel='')
            axs[j, k].text(0.05, 0.99, rf'$({chr(97 + k + 2 * j)})$ {labels[j][k]}',
                           ha='left', va='top', c='white', transform=axs[j, k].transAxes)
            axs[j, k].tick_params(color='white')
    for j in range(2):
        axs[1, j].set_xlabel(r'$\rm{Re}(\alpha)$')
        axs[j, 0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.tight_layout(pad=0.15)
    fig.savefig(local_plot_path(__file__) / 'fig1.pdf')


def fig2():
    """Separation in the Liouvillian spectrum.

    Requires to execute `ev.py` with the arguments:
        -n 4
        -m 4
        --g2min 0.01
        --g2max 0.2
        --g2num 200
        --emin 0.0
        --emax 5.0
        --enum 200
        --D 0.4
    """
    latexify(plt, type='paper', fract=0.2, palette='qualitative')

    fpath = local_data_path('ev', 4, 4) / 'g1&1_g2s&0.01&0.2&200_etas&0.0&5.0&200_D&0.4_dim&50.npy'

    fig, ax = plt.subplots()

    _, _, Cs = plot_gap(fpath, fig=fig, ax=ax, levels=[0.01, 0.1, 0.5], colors=['C1', 'C2', 'C0'])

    ax.clabel(Cs, inline=True,
              manual=[(0.1, 2), (0.1, 1.11), (0.12, 0.2)],
              fmt=lambda l: rf'$\tau_4 = {int(1/l):d}$')

    ax.text(0.5, 0.99, r'$n = m = 4$', ha='center', va='top', transform=ax.transAxes, c='w')

    fig.savefig(local_plot_path(__file__) / 'fig2.pdf')


def fig3():
    """Example time evolution of an initial state."""
    latexify(plt, type='paper', fract=0.2, palette='qualitative')

    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'hspace': 0.02})

    ns = [3, 4]
    taus = [2, 10, 100]
    colors = ['C1', 'C2', 'C0']

    k = 0
    D = 0.4
    for ax, n in zip(axs, ns):
        j = 0
        for g2, eta, dim in POINTS[n]:
            c = colors[j]
            ev = eigvals(g2, eta, D, n, n, dim)

            fpath = local_data_path('evolution', n, n) / f'g-{g2}_e-{eta}_d-{D}_t-0-4999.999999999999-50000.npy'

            if not fpath.exists():
                continue

            data = np.load(fpath)
            ax.plot(np.real(data[:, 0]), np.abs(data[:, 1]), c=c)

            t0 = -1 / np.real(ev[n])
            t1 = -1 / np.real(ev[n - 1])

            if n == 4:
                ymin, ymax = min(0.5 * j, 0.75), 0.5 + 0.25 * j
                ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, color=c, alpha=0.2, ec=None)
            else:
                ymin, ymax = j / 3, (j + 1) / 3
                ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, color=c, alpha=0.2, ec=None)

            ax.text(0.2 + 0.2 * j,
                    ymin * 0.85 + ymax * 0.15 if j == 0 else ymin * 0.95 + ymax * 0.05,
                    rf'$\tau_{n} = {taus[j]}$',
                    c=c, ha='center', va='bottom', transform=ax.transAxes)
            j += 1

        ax.text(1, 0.99, rf'$({chr(97 + k)})\ n = {n}$', ha='right', va='top', transform=ax.transAxes)
        ax.set(
            xlabel=r'$\gamma_1 t$' if k > 0 else '',
            ylabel=r'$|\langle a \rangle|$',
            xscale='log')
        ax.set_xlim(left=0.01)

        k += 1

    fig.savefig(local_plot_path(__file__) / 'fig3.pdf')


def fig4():
    """First version of Fig. 4 containing success probability in panel (a) and storage capacity in (b)."""
    latexify(plt, type='paper', fract=0.35, palette='qualitative')

    fig, axs = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.55, 0.45], 'hspace': 0.25})

    # plot_time_only_n(4, fig=fig, ax=axs[0])
    simple_plot([2, 3, 4, 5], fig=fig, ax=axs[1])

    # handles, labels = axs[1].get_legend_handles_labels()
    # axs[1].legend(handles, labels, ncol=2, loc='upper center', title=r'$n$')
    axs[1].get_legend().remove()

    ax_inset = axs[1].inset_axes(bounds=[0.45, 0.32, 0.45, 0.65], transform=axs[1].transAxes)
    max_storage_capacity([2, 3, 4, 5, 6], fig=fig, ax=ax_inset)
    ax_inset.get_legend().remove()

    axs[0].text(0.99, 0.98, r'$(a)$', transform=axs[0].transAxes, ha='right', va='top')
    axs[1].text(0.99, 0.98, r'$(b)$', transform=axs[1].transAxes, ha='right', va='top')

    fig.savefig(local_plot_path(__file__) / 'fig4.pdf')


def fig4b():
    """Second version of Fig. 4 including system size vs. number of patterns in panel (c)."""
    latexify(plt, type='paper', fract=0.28, palette='qualitative')

    fig, axs = plt.subplot_mosaic([['a', 'a'], ['b', 'c']], gridspec_kw={
        'height_ratios': [0.5, 0.5], 'hspace': 0.29, 'width_ratios': [0.58, 0.42], 'wspace': 0.26
    })

    # Panel A

    memorypath = local_data_path('memory', 4, 4)
    dims = [40, 30, 20, 15]
    files = [memorypath / f'1_0.1_1.1423118881998557_0.4_{dim}' for dim in dims]

    plot_memory(4, files, fig=fig, ax=axs['a'], labels=dims)
    axs['a'].legend(loc='lower left', bbox_to_anchor=(0.05, 0.06), title=r'$\dim \mathcal{H}_{eff}$', ncol=2)

    # Panel B

    simple_plot([2, 3, 4, 5], fig=fig, ax=axs['b'], show_top=True)

    handles, labels = axs['b'].get_legend_handles_labels()
    handles = [Line2D([], [], c=f'C{j}', marker=MARKERS[j]) for j in range(4)]
    axs['b'].set_ylim(top=0.5)
    axs['b'].legend(handles, labels, ncol=2, loc='upper left', bbox_to_anchor=(0.31, 1.), title=r'$n$')

    # Panel C

    max_storage_capacity([2, 3, 4, 5], fig=fig, ax=axs['c'])
    axs['c'].get_legend().remove()
    axs['c'].text(3, 3.5 / 0.138, 'Hebbian', ha='center', va='center', rotation=43)

    axs['a'].text(0.99, 0.98, r'$(a)$', transform=axs['a'].transAxes, ha='right', va='top')
    axs['b'].text(0.99, 0.98, r'$(b)$', transform=axs['b'].transAxes, ha='right', va='top')
    axs['c'].text(0.02, 0.98, r'$(c)$', transform=axs['c'].transAxes, ha='left', va='top')

    fig.savefig(local_plot_path(__file__) / 'fig4b.pdf')


if __name__ == '__main__':
    # fig1()
    # fig2()
    fig3()
    # fig4b()
