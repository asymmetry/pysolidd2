#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants, integrate

__all__ = ['f1p', 'f2p', 'g1p', 'g2p']

_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3


def _r(x, q2):
    a = [0, 0.0485, 0.5470, 2.0621, -0.3804, 0.5090, -0.0285]
    b = [0, 0.0481, 0.6114, -0.3509, -0.4611, 0.7172, -0.0317]
    c = [0, 0.0577, 0.4644, 1.8288, 12.3708, -43.1043, 41.7415]

    theta = 1 + 12 * q2 / (q2 + 1) * 0.125**2 / (0.125**2 + x**2)

    ra = a[1] / numpy.log(q2 / 0.04) * theta + a[2] / (q2**4 + a[3]**4)**0.25 * (1 + a[4] * x + a[5] * x**2) * x**a[6]
    rb = b[1] / numpy.log(q2 / 0.04) * theta + (b[2] / q2 + b[3] / (q2**2 + 0.3**2)) * (1 + b[4] * x + b[5] * x**2) * x**b[6]
    q2thr = c[4] * x + c[5] * x**2 + c[6] * x**3
    rc = c[1] / numpy.log(q2 / 0.04) * theta + c[2] / ((q2 - q2thr)**2 + c[3]**2)**0.5

    return (ra + rb + rc) / 3


def f1p(x, q2):
    return f2p(x, q2) * (1 + 4 * _m_p**2 * x**2 / q2) / (2 * x * (1 + _r(x, q2)))


def f2p(x, q2):
    # NMC
    # Phys. Lett. B364(1995)107
    a = [0, -0.02778, 2.926, 1.0362, -1.840, 8.123, -13.074, 6.215]
    b = [0, 0.285, -2.694, 0.0188, 0.0274]
    c = [0, -1.413, 9.366, -37.79, 47.10]

    ax = x**a[1] * (1 - x)**a[2] * (a[3] + a[4] * (1 - x) + a[5] * (1 - x)**2 + a[6] * (1 - x)**3 + a[7] * (1 - x)**4)
    bx = b[1] + b[2] * x + b[3] / (x + b[4])
    cx = c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4

    gamma2 = 0.25**2

    return ax * (numpy.log(q2 / gamma2) / numpy.log(20 / gamma2))**bx * (1 + cx / q2)


def g1p(x, q2):
    # SLAC E155
    # Phys. Lett. B493(2000)19, Eq.(5)
    return x**0.700 * (0.817 + 1.014 * x - 1.489 * x**2) * (1 - 0.04 / q2) * f1p(x, q2)


def g2p(x, q2, ww=True):
    # only have g2ww at this moment
    if numpy.isscalar(x):
        if x > 0:
            result = -g1p(x, q2) + integrate.quad(lambda y: g1p(y, q2) / y, x, 1)
        else:
            result = numpy.inf
    else:
        result = numpy.empty_like(x)
        for (xx, rr) in numpy.nditer([x, result], [], [['readonly'], ['writeonly']]):
            if xx > 0:
                rr[...] = -g1p(xx, q2) + integrate.quad(lambda y: g1p(y, q2) / y, xx, 1)[0]
            else:
                rr = numpy.inf

    return result
