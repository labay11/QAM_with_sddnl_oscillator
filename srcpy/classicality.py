import os
import numpy as np
from argparse import ArgumentParser
from joblib import Parallel, delayed

from model import classicallity
from ems import metastable_states
from utils import DATA_PATH, driving_dissipation_ratio


def _internal(beta, g2, d, nl_eta, nl_dis):
    eta = driving_dissipation_ratio(beta, nl_eta, nl_dis) * g2
    dim = max(min(140, int(beta) * 3), 50)
    ems, ls, rs, ps = metastable_states(g2, eta, d, nl_eta, nl_dis, m=dim)
    _, c, _ = classicallity(ls, ems[1:])
    return c


def compute_classicality(betas, nl_eta, nl_dis, g2=0.4, d=0.4, parallel=True):
    if parallel:
        res = Parallel(n_jobs=4)(delayed(_internal)(beta, g2, d, nl_eta, nl_dis) for beta in betas)
    else:
        res = [_internal(beta, g2, d, nl_eta, nl_dis) for beta in betas]

    return np.array(res)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-nl', type=int, help='nl_eta == nl_dis')
    parser.add_argument('-ml', type=int, help='method to use (1: me, 2: deterministic)')
    parser.add_argument('-p', '--plot', action='store_true')

    args = parser.parse_args()

    fpath = DATA_PATH / 'classicallity/a'
    betas = np.linspace(0.1, 10, 11)
    if args.plot:
        import matplotlib.pyplot as plt
        for f in os.listdir(fpath):
            C = np.load(os.path.join(fpath, f))
            plt.plot(betas[1:], np.log10(C)[1:], label=f)
        plt.legend()
        plt.show()
    else:
        C = compute_classicality(betas, args.nl, args.ml)

        os.makedirs(fpath, exist_ok=True)
        np.save(os.path.join(fpath, f'{args.nl}_{args.ml}.npy'), C)
