import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from utils import latexify, local_data_path, local_plot_path, build_filename


def fig_q():
    latexify(plt, type='preprint', fract=0.15)

    def ls(n, m):
        if m == n - 1:
            return ':'
        elif m == n + 1:
            return '--'
        else:
            return '-'

    fig, axs = plt.subplots(ncols=3, gridspec_kw={'wspace': 0.15})

    dirpath = local_data_path('squeezing_ss')
    mnms = sorted(
        [tuple(map(int, folder.name.split('_'))) for folder in dirpath.iterdir()],
        key=lambda x: (x[0], x[1])
    )

    X = np.linspace(0.0001, 6, 100)
    fname = build_filename(g2=0.2, D=0.4, betas=X) + '.npy'

    for n_ in [2, 3, 4]:
        fpaths = [(dirpath / f'{n}_{m}' / fname, m) for n, m in mnms if n == n_]

        j = 0
        for file, m in fpaths:
            if not file.exists():
                continue
            ev = np.load(file)
            sq = (ev[:, 1] - ev[:, 0]**2) / ev[:, 0] - 1
            axs[n_ - 2].plot(X, np.real(sq), label=str(m), ls=ls(n_, m), c=f'C{j}')
            j += 1

        axs[n_ - 2].text(0.99, 0.98, rf'$({chr(97 + n_ - 2)})\ n = {n_}$',
                         va='top', ha='right', transform=axs[n_ - 2].transAxes)
        # axs[n_ - 2].legend(title=r'$m$')

    for ax in axs:
        ax.axhline(0.0, c='k')
        ax.set_xlabel(r'$R$')
    axs[0].set_ylabel(r'$\mathcal{Q}$')

    fig.savefig(local_plot_path(__file__) / 'q_mandel_ss.pdf')


fig_q()
