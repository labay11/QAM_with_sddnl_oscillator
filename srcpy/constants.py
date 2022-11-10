from collections import OrderedDict

from utils import driving_dissipation_ratio

DELTA = 0.4
G1 = 1.0
DIM = 50
DIM_SS = 120

POINTS = {
    3: [
        (0.8, 0.47412171491586075, 40),
        (0.6, 1.3755952741694344, 50),
        (0.4, 1.984379582650444, 50)  # , (0.3, 2, 40)
    ],
    4: [
        (0.15, 0.5684021044214842, 40),
        (0.1, 1.1423118881998557, 40),
        (0.05, 1.4210869697181017, 50)  # , (0.05, 2, 40)
    ]
}

SQUEEZING = OrderedDict([
    ((2, 1), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': 0.3,
        'D': DELTA
    }),
    ((2, 2), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': driving_dissipation_ratio(3.0, 2, 2) * 0.2,
        'D': DELTA
    }),
    ((2, 3), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': driving_dissipation_ratio(3.5, 2, 3) * 0.2,
        'D': DELTA
    }),
    ((3, 2), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': 0.32,  # driving_dissipation_ratio(3.65, 3, 2) * 0.2,
        'D': DELTA
    }),
    ((3, 3), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': driving_dissipation_ratio(3.0, 3, 3) * 0.2,
        'D': DELTA
    }),
    ((3, 4), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': driving_dissipation_ratio(3.65, 3, 4) * 0.2,
        'D': DELTA
    }),
    ((4, 3), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': 1.5,  # driving_dissipation_ratio(1.78, 4, 3) * 0.2,
        'D': DELTA
    }),
    ((4, 4), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': driving_dissipation_ratio(3.0, 4, 4) * 0.2,
        'D': DELTA
    }),
    ((4, 5), {
        'g1': 1.0,
        'g2': 0.2,
        'eta': driving_dissipation_ratio(4.0, 4, 5) * 0.2,
        'D': DELTA
    })
])
