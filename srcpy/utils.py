from pathlib import Path
import numpy as np

from cycler import cycler

DATA_PATH = Path('~/Documents/data/gqvdp/').expanduser()
PLOT_PATH = Path('~/Documents/GitHub/QAM_with_sddnl_oscillator/plots/').expanduser()
TOL = 1e-20


def local_data_path(fname, nl_eta=None, nl_dis=None):
    path = DATA_PATH / Path(fname).stem
    if nl_eta is not None and nl_dis is not None:
        path = path / f'{nl_eta}_{nl_dis}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_plot_path(fname, nl_eta=None, nl_dis=None):
    path = PLOT_PATH / Path(fname).stem
    if nl_eta is not None and nl_dis is not None:
        path = path / f'{nl_eta}_{nl_dis}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def amplitude(g2, eta, n, m):
    if n == 2 * m:
        return np.power(1 / (m * (4 * eta - g2)), 1 / (2 * m - 2))

    return np.power(2 * n * eta / (m * g2), 1 / (2 * m - n))


def driving_dissipation_ratio(amplitude, nl_eta, nl_dis):
    return np.power(amplitude, 2 * nl_dis - nl_eta) * nl_dis / (2 * nl_eta)


MARKERS = ['o', 's', '^', 'v', '<', '>', 'd']
LINE_STYLES = ['-', '--', ':', '-.']
COLORS = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB']

COLOR_CYCLES = {
    'default': cycler('color', COLORS),
    'default_mk': (cycler('color', ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB']) +
                   cycler('marker', ['o', 's', '^', 'v', '<', '>', 'd'])),
    'qualitative': cycler('color',
                          ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',  '#FFFF33', '#A65628', '#F781BF']),
    'diverging': cycler('color',
                        ['#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F']),
    'sequential': cycler('color',
                         ['#0C2C84', '#225EA8', '#1D91C0', '#41B6C4', '#7FCDBB', '#C7E9B4', '#EDF8B1', '#FFFFD9'])
}


PAPER_TYPES = {
    'def': (426., 672.),  # change 600 to actual value
    'preprint': (510., 672.),
    'paper': (510. / 2, 672.),
    'beamer169': (398.3386, 243.76566),
    'beamer43': (357.77, 260.83748)
}


GOLDEN_MEAN = (np.sqrt(5) - 1.0) / 2.0
INCHES_PER_PT = 1.0 / 72.27  # Convert pt to inch


def latex_figsize(type, fract=None):
    if fract is None:
        wf, hf = 1., GOLDEN_MEAN
    elif isinstance(fract, (tuple, list)):
        wf, hf = fract  # will raise if dif than two elements
    else:
        wf, hf = 1., fract

    pw, ph = PAPER_TYPES[type]
    fig_width = pw * wf * INCHES_PER_PT  # width in inches
    fig_height = fig_width * hf if fract is None else ph * hf * INCHES_PER_PT
    return [fig_width, fig_height]


DEFAULT_PARAMS = {
    'backend': 'ps',
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 10,

    'xtick.labelsize': 7,
    'xtick.direction': 'in',
    'xtick.major.size': 3,
    'xtick.major.width': 0.5,
    'xtick.minor.size': 1.5,
    'xtick.minor.width': 0.5,
    'xtick.minor.visible': True,
    'xtick.top': False,

    'ytick.labelsize': 7,
    'ytick.direction': 'in',
    'ytick.major.size': 3,
    'ytick.major.width': 0.5,
    'ytick.minor.size': 1.5,
    'ytick.minor.width': 0.5,
    'ytick.minor.visible': True,

    'axes.xmargin': 0.02,
    'axes.ymargin': 0.02,
    'lines.linewidth': 0.7,
    'lines.markersize': 3,
    'text.usetex': True,
    'figure.figsize': latex_figsize('def', None),
    'axes.grid': False,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'axes.labelpad': 1.0,
    'axes.prop_cycle': COLOR_CYCLES['default'],
    "axes.formatter.use_mathtext": True,
    'errorbar.capsize': 5,
    # Always save as 'tight'
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': '0.01',  # Use virtually all space when we specify figure dimensions
    'figure.dpi': 500.0
}


def latexify(pyplot, type='def', fract=None, palette='default', **kwargs):
    """Latexify the passed matplotlib instance.

    Sets the default figure size to follow the golden ratio, the default width in points is the `columnwidth`
    in LaTeX. Also reduces the line widths, sets other default line colors and most importantly,
    uses latex font styles.

    Parameters
    ----------
    plt : matplotlib.pytplot
        the pyplot instance
    fract : float, optional
        height fraction of the plot with respect to the width, default to 1.
    By default the height is calculated based on the golden ration,
        to reduce the height simply pass a value < 1.

    Notes
    -----
    It is necessary that you have LaTeX installed in your computer.
    """
    params = DEFAULT_PARAMS.copy()
    params.update({
        'figure.figsize': latex_figsize(type, fract),
        'axes.prop_cycle': COLOR_CYCLES[palette] if palette in COLOR_CYCLES else palette,
        **kwargs
    })
    pyplot.rcParams.update(params)