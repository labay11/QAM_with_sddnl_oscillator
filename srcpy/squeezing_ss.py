from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import qutip as qt

from model import build_system
from utils import local_data_path, local_plot_path, latexify,\
    driving_dissipation_ratio, build_filename, amplitude, parse_filename


def ss_to_ems(ss, n, beta):
    xlim = int(beta**3)
    xvec = np.linspace(-xlim, xlim, 10**(max(2, int(np.log10(xlim)))))
    L2 = len(xvec) // 2

    W = qt.wigner(ss, xvec)

    if n == 2:
        W[:, L2:] = 0.0
        W *= 2
    elif n == 3:
        for j in range(L2):
            W[j, j:] = 0.0
            W[-j, j:] = 0.0
        W *= 3
    elif n == 4:
        W[:, :L2] = 0.0
        W[:L2, L2:] = 0.0
        W *= 4

    return W, xvec


def wexpect(wop, w, xvec):
    dx = (xvec[-1] - xvec[0]) / len(xvec)
    A = w * wop
    return trapezoid(trapezoid(A, dx=dx), dx=dx)


def squeezing_amplitude(g2, betas, D, n, m):
    N = len(betas)

    EV = np.zeros((N, 4), dtype=complex)

    for j in range(1, N):
        eta = driving_dissipation_ratio(betas[j], n, m) * g2
        dim = min(max(int(round(3 * betas[j]**3)), 100), 200)
        num = qt.num(dim)
        num2 = num**2
        try:
            L = build_system(1.0, g2, eta, D, n, m, dim)
            ss = qt.steadystate(L)
            EV[j, 0] = qt.expect(num, ss)
            EV[j, 1] = qt.expect(num2, ss)
            print(j, dim, EV[j], ss.isoper, ss.isherm, ss.tr())
        except Exception as e:
            print(j, e)
            EV[j, :] = np.nan

        try:
            w, xvec = ss_to_ems(ss, n, betas[j])
            opa = qt.destroy(dim)
            opa2 = opa**2

            wopa = qt.wigner(opa, xvec)
            wopa2 = qt.wigner(opa2, xvec)

            EV[j, 2] = wexpect(wopa, ss, xvec)
            EV[j, 3] = wexpect(wopa2, ss, xvec)
        except Exception as e:
            print(j, 'wexpect', e)
            EV[j, 2:] = np.nan

    np.save(local_data_path(__file__, n, m) / build_filename(g2=g2, D=D, betas=betas, ext='.npy'), EV)


def squeezing_driving(g2, etas, D, n, m):
    N = len(betas)

    EV = np.zeros((N, 4, n), dtype=complex)

    for j in range(N):
        beta = amplitude(g2, etas[j], n, m)
        dim = min(max(int(round(3 * beta**3)), 100), 200)
        opa = qt.destroy(dim)
        opa2 = opa*opa
        num = qt.num(dim)
        num2 = num*num
        try:
            L = build_system(1.0, g2, etas[j], D, n, m, dim)
            ss = qt.steadystate(L)
            EV[j, 0] = qt.expect(num, ss)
            EV[j, 1] = qt.expect(num2, ss)
            EV[j, 2] = qt.expect(opa, ss)
            EV[j, 3] = qt.expect(opa2, ss)
        except Exception as e:
            print(j, e)
            EV[j, :, :] = np.nan

    np.save(local_data_path(__file__, n, m) / build_filename(g2=g2, D=D, etas=etas, ext='.npy'), EV)


def plot_lines(files, outpath, mnms, labels=None, right_amp=False):
    if not files:
        return

    latexify(plt, type='paper', fract=(0.5))

    if labels is None:
        labels = [None] * len(files)

    fig, ax = plt.subplots(nrows=3, sharex=True)

    i_min = 1

    for file, lbl, (n, m) in zip(files, labels, mnms):
        ev = np.load(file)[i_min:]
        params = parse_filename(file.stem)
        betas = params['betas'] if 'betas' in params else params['etas']
        # sq = np.sqrt(ev[:, 1] - ev[:, 0]**2) / ev[:, 0]
        sq = (ev[:, 1] - ev[:, 0]**2) / ev[:, 0] - 1
        print(sq.shape, betas.shape, ev.shape)
        sq[np.abs(sq) > 10] = np.nan
        ax[0].plot(betas[i_min:], np.real(sq), label=lbl)

        Da2 = ev[:, 3] - ev[:, 2]**2
        Dad2 = np.conj(ev[:, 3]) - np.conj(ev[:, 2])**2
        Dn = ev[:, 0] - np.abs(ev[:, 2])**2

        X1 = (Da2 + Dad2 + 2 * Dn + 1)*0.25
        X2 = (-Da2 - Dad2 + 2 * Dn + 1)*0.25
        X1[np.abs(X1) > 100] = np.nan
        X2[np.abs(X2) > 100] = np.nan
        ax[1].plot(betas[i_min:], np.real(X1), label=lbl)
        ax[2].plot(betas[i_min:], np.real(X2), label=lbl)

    ax[0].axhline(0.0, c='k', ls='-')
    ax[1].axhline(0.25, c='k', ls='-')
    ax[2].axhline(0.25, c='k', ls='-')

    ax[0].set(ylabel='Squeezing')
    ax[1].set(ylabel=r'$(\Delta X_1)^2$')
    ax[2].set(xlabel=r'$R$', ylabel=r'$(\Delta X_2)^2$')
    ax[0].legend(ncol=2)

    fig.savefig(outpath)


def plot_all():
    dirpath = local_data_path(__file__)

    for folder in dirpath.iterdir():
        n, m = map(int, folder.name.split('_'))
        savepath = local_plot_path(__file__, n, m)
        for file in folder.iterdir():
            if file.suffix != '.npy' or file.is_dir():
                continue

            if file.name.startswith('beta'):
                # single line plot
                plot_lines([file], savepath / f'{file.stem}.pdf')
            else:
                continue


def plot_comparison(n, m, filter_by, outpath, label):
    dirpath = local_data_path(__file__, n, m)
    files = [
        f
        for f in dirpath.iterdir()
        if f.suffix == '.npy' and not f.is_dir() and f.name.startswith('beta') and filter_by(parse_filename(f.stem))
    ]

    plot_lines(files, outpath, [label(parse_filename(f.stem)) for f in files])


def plot_gd_comparisons():
    ds = [0.1, 0.2, 0.5, 1.5]
    gs = [0.1, 0.2, 0.4, 0.8, 1.2]

    dirpath = local_data_path(__file__)

    for folder in dirpath.iterdir():
        n, m = map(int, folder.name.split('_'))

        for d in ds:
            plot_comparison(
                n, m,
                lambda x: x[2] == d,
                local_plot_path(__file__, n, m) / f'compare_d-{d}.pdf',
                lambda x: x[1])

        for g in gs:
            plot_comparison(
                n, m,
                lambda x: x[1] == g,
                local_plot_path(__file__, n, m) / f'compare_g-{g}.pdf',
                lambda x: x[2])


def plot_nm_comparisons(g, betas, D, driving=False):
    dirpath = local_data_path(__file__)
    outpath = local_plot_path(__file__, 'n', 'm')

    mnms = sorted(
        [tuple(map(int, folder.name.split('_'))) for folder in dirpath.iterdir()],
        key=lambda x: (x[0], x[1])
    )
    labels = [f'${n} - {m}$' for n, m in mnms]

    fname = build_filename(g2=g, D=D, betas=betas) if not driving else build_filename(g2=g, D=D, etas=betas)
    fname += '.npy'

    fpaths = [dirpath / f'{n}_{m}' / fname for n, m in mnms]
    fpaths = [f for f in fpaths if f.exists()]
    if not fpaths:
        return

    plot_lines(fpaths, outpath / (fname[:-3] + 'pdf'), mnms, labels)
    for n_ in [2, 3, 4]:
        fpaths = [dirpath / f'{n}_{m}' / fname for n, m in mnms if n == n_]
        fpaths = [f for f in fpaths if f.exists()]
        plot_lines(fpaths, outpath / (f'n&{n_}_' + fname[:-3] + 'pdf'), mnms,
                   [f'${n} - {m}$' for n, m in mnms if n == n_])


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-g', type=float, help='g2', default=0.2)
    parser.add_argument('-d', type=float, help='delta', default=0.4)
    parser.add_argument('-n', type=int, help='nl_eta == nl_dis')
    parser.add_argument('-m', type=int, help='method to use (1: me, 2: deterministic)')
    parser.add_argument('--bmin', type=float, help='beta min', default=0.0001)
    parser.add_argument('--bmax', type=float, help='beta max')
    parser.add_argument('--bnum', type=int, help='beta num')
    parser.add_argument('-e', action='store_true')

    parser.add_argument('-p', '--plot', action='store_true')

    args = parser.parse_args()

    betas = np.linspace(args.bmin, args.bmax, args.bnum)
    if args.plot:
        plot_nm_comparisons(args.g, betas, args.d, args.e)
    else:
        if args.e:
            squeezing_driving(args.g, betas, args.d, args.n, args.m)
        else:
            squeezing_amplitude(args.g, betas, args.d, args.n, args.m)
