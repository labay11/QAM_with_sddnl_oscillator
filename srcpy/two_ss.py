from pathlib import Path
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from wigner import plot_multiple_wigner
from utils import PLOT_PATH, latexify, amplitude

latexify(plt)


def local_data_path(nl_eta, nl_dis):
    path = PLOT_PATH / Path(__file__).stem / f'{nl_eta}_{nl_dis}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalise_eigvecs(evec):
    op = qt.vector_to_operator(evec)
    op = (op + op.dag()) * 0.5
    return op / op.tr()


def plot_raw_modes(g2, eta, D, nl_eta, nl_dis, dim):
    L = build_system(1, g2, eta, D, nl_eta=nl_eta, nl_dis=nl_dis, dim=dim)
    ev, eigvecsr = L.eigenstates(sort='high', eigvals=2)
    beta = amplitude(g2, eta, nl_eta, nl_dis)

    fig, axs = plt.subplots(ncols=2, sharey=True)
    plot_multiple_wigner(fig, axs, [qt.vector_to_operator(evr) for evr in eigvecsr], alpha_max=beta*1.75)
    axs[1].set_ylabel('')
    for lamb, ax in zip(ev, axs):
        ax.set_title("{num.real:+0.04g} {num.imag:+0.04g}j".format(num=lamb))

    fig.savefig(local_data_path(nl_eta, nl_dis) / f'{g2}_{eta}_{D}.pdf')


for g2 in [0.2, 0.4, 0.6, 0.8]:
    for eta in [1, 3, 5]:
        for D in [0.4, 0.8, 1.2]:
            plot_raw_modes(g2, eta, D, 4, 3, 80)
