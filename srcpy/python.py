import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from utils import latexify

NODES_TO_COLOR = ['m', 'b', 'r', 'g']
PROTOCOL_TO_MARKER = ['o', 's', '^', 'v']


def add_legend_to_fig(fig, show_as_repeaters, adjust):
    nodes_keys = [2, 5, 9, 17]
    if show_as_repeaters:
        nodes_names = map(lambda x: x - 2, nodes_keys.copy())
    else:
        nodes_names = nodes_keys.copy()

    lines_nodes = [
        Line2D([0], [0], lw=2, color=NODES_TO_COLOR[j])
        for j in range(len(nodes_keys))
    ]
    labels_nodes = list(map(str, nodes_names))
    nodes_legend = fig.legend(
        lines_nodes, labels_nodes,
        title='Number of repeaters', loc='upper right', ncol=4, bbox_to_anchor=(0.55, 1.)
    )
    plt.gca().add_artist(nodes_legend)
    markers = [
        (Line2D([0], [0], mec='k', mfc='None', mew=0.7, ls='None', marker=mk),
         Line2D([0], [0], color='k', ls='None', marker=mk))
        for mk in PROTOCOL_TO_MARKER
    ]
    protocol_legend = fig.legend(
        markers, ['SWAP-ASAP', 'EPL', 'DEJMPS (1)', 'DEJMPS (2)'], numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        title='Strategies', loc='upper left', ncol=2, bbox_to_anchor=(0.58, 1.)
    )
    plt.gca().add_artist(protocol_legend)
    markers = [
        tuple([Line2D([0], [0], mec='k', mfc='None', mew=0.7, ls='None', marker=n) for n in PROTOCOL_TO_MARKER]),
        tuple([Line2D([0], [0], mec='k', mfc='k', ls='None', marker=n) for n in PROTOCOL_TO_MARKER])
    ]
    heg_legend = fig.legend(
        markers, ['SC', 'DC'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
        title='HEG', loc='upper left', ncol=1, bbox_to_anchor=(0.4, 0.8)
    )

    fig.subplots_adjust(top=adjust)


if __name__ == '__main__':
    latexify(plt, type='paper')
    fig, ax = plt.subplots()
    add_legend_to_fig(fig, True, 0.2)
    fig.savefig('legend.pdf')
