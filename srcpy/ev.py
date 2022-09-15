import numpy as np
import os
import argparse

from model import build_system
from utils import local_data_path, amplitude, build_filename


def ev(n_eigvals, **params):
    if 'dim' not in params:
        params['dim'] = min(max(50, round(amplitude(**params)) * 3), 120)

    L = build_system(**params)
    return L.eigenenergies(sort='high', eigvals=n_eigvals)


def ev_matrix(n_eigvals, g2s, etas, **other_params):
    Ng, Ne = len(g2s), len(etas)
    fname = build_filename(g2s=g2s, etas=etas, **other_params)

    EV = np.zeros((Ng, Ne, n_eigvals), dtype=complex)
    for i, j in np.ndindex(Ng, Ne):
        try:
            EV[i, j, :] = ev(n_eigvals, g2=g2s[i], eta=etas[j], **other_params)
        except Exception as e:
            print(f'Error calculating {i} (g2 = {g2s[i]}):', e)
            EV[i, j, :] = np.nan

    dirpath = local_data_path(__file__, other_params['n'], other_params['m'])
    np.save(dirpath / (fname + '.npy'), EV)


def ev_matrix_row(j, n_eigvals, g2s, etas, **other_params):
    Ng = len(g2s)
    fname = build_filename(g2s=g2s, etas=etas, **other_params)

    EV = np.zeros((Ng, n_eigvals), dtype=complex)
    dirpath = local_data_path(__file__, other_params['n'], other_params['m']) / fname
    os.makedirs(dirpath, exist_ok=True)
    #
    # if os.path.exists(os.path.join(dirpath, f'{j}.npy')):
    #     print('Already exists')
    #     return 0

    for i in range(Ng):
        try:
            EV[i, :] = ev(n_eigvals, g2=g2s[i], eta=etas[j], **other_params)
        except Exception as e:
            print(f'Error calculating {i} (g2 = {g2s[i]}):', e)
            EV[i, :] = np.nan

    np.save(os.path.join(dirpath, f'{j}.npy'), EV)


def ev_matrix_point(j, k, n_eigvals, g2s, etas, **other_params):
    fname = build_filename(g2s=g2s, etas=etas, **other_params)

    EV = ev(n_eigvals, g2=g2s[k], eta=etas[j], **other_params)
    EV = EV.reshape(1, n_eigvals)

    dirpath = local_data_path(__file__, other_params['n'], other_params['m']) / fname
    os.makedirs(dirpath, exist_ok=True)
    # os.makedirs(dirpath, exist_ok=True)
    np.save(os.path.join(dirpath, f'{j}_{k}.npy'), EV)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-r', type=int, help='row', default=None)
    parser.add_argument('-c', type=int, help='column', default=None)
    parser.add_argument('-n', type=int, help='nl_eta')
    parser.add_argument('-m', type=int, help='nl_dis')
    parser.add_argument('--D', default=0.4, type=float, help='delta')
    parser.add_argument('--g1', default=1., type=float, help='gamma 1')
    parser.add_argument('--g2min', type=float, default=0.)
    parser.add_argument('--g2max', type=float, default=1.)
    parser.add_argument('--g2num', type=int, default=100)
    parser.add_argument('--emin', type=float, default=0.)
    parser.add_argument('--emax', type=float, default=5.)
    parser.add_argument('--enum', type=int, default=100)
    parser.add_argument('--nev', type=int, default=8)

    args = parser.parse_args()

    g2s = np.linspace(args.g2min, args.g2max, args.g2num)
    etas = np.linspace(args.emin, args.emax, args.enum)

    if args.c is None:
        if args.r is None:
            ev_matrix(args.nev, g2s, etas, **vars(args))
        else:
            ev_matrix_row(args.r, args.nev, g2s, etas, **vars(args))
    else:
        ev_matrix_point(args.r, args.c, args.nev, g2s, etas, **vars(args))
