import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import qutip as qt

from storing_capacity import simple_plot, max_storage_capacity
from wigner import plot_multiple_wigner
from utils import latexify, local_plot_path, local_data_path, parse_filename, amplitude, MARKERS


def fig4b():
    """Second version of Fig. 4 including system size vs. number of patterns in panel (c)."""
    latexify(plt, type='beamer43', fract=0.6, palette='qualitative')

    fig, axs = plt.subplot_mosaic("bc", gridspec_kw={'width_ratios': [0.6, 0.4], 'wspace': 0.2})

    # Panel B

    simple_plot([2, 3, 4, 5], fig=fig, ax=axs['b'], show_top=True, show_unbounded=False)

    handles, labels = axs['b'].get_legend_handles_labels()
    handles = [Line2D([], [], c=f'C{j}', marker=MARKERS[j]) for j in range(4)]
    # axs['b'].set_ylim(top=0.5)
    axs['b'].legend(handles, labels, ncol=2, loc='center right', bbox_to_anchor=(0.95, 0.75), title=r'$n$')

    # Panel C

    max_storage_capacity([2, 3, 4, 5], fig=fig, ax=axs['c'])
    axs['c'].get_legend().remove()
    axs['c'].text(3, 3.2 / 0.138, 'Hebbian', ha='center', va='center', rotation=48)

    axs['b'].text(0.99, 0.98, r'$(a)$', transform=axs['b'].transAxes, ha='right', va='top')
    axs['c'].text(0.02, 0.98, r'$(b)$', transform=axs['c'].transAxes, ha='left', va='top')

    fig.savefig(local_plot_path(__file__) / 'fig4b.pdf')


def wigner_trajectory(n, fname, indices, title_times=None, show_amplitude=True):
    latexify(plt, 'beamer43', fract=0.3)

    fpath = local_data_path('trajectory', n, n) / fname
    data = qt.qload(str(fpath))

    params = parse_filename(fname)
    beta = amplitude(n=n, m=n, **params) * 1.35
    lobes = np.array([beta * np.exp(1j * (2 * j + 1) * np.pi / n) for j in range(4)])

    states = np.mean(data.states, axis=0) if fname.startswith('mc') else data.states
    times = data.times

    fig, axs = plt.subplots(ncols=len(indices), sharey=True, gridspec_kw={'wspace': 0.01})
    plot_multiple_wigner(fig, axs, [states[j] for j in indices])

    if title_times is None:
        title_times = [f'{times[j]:0.2f}' for j in indices]

    for ax, title in zip(axs, title_times):
        ax.set_ylabel('')
        ax.text(0.5, 0.99, rf'$\gamma_1 t = {title}$', color='w', transform=ax.transAxes, va='top', ha='center', fontsize=6)

        if show_amplitude:
            ax.scatter(np.real(lobes), np.imag(lobes), marker='x', color='k')

    axs[0].set_ylabel(r'$\rm{Im}(\alpha)$')

    fig.savefig(local_plot_path(__file__) / (fname + '.pdf'))


wigner_trajectory(4, 'mc&1665580589898012422_g1&1.0_g2&0.1_eta&1.1423118881998557_D&0.4_dim&40',
                  [0, 1000, 2500, 4600, 4999], ['0', '0.06', r'1', '50', r'100'])
