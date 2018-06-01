#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants

from .cross_section import dxslp_slac, dxstp_slac, xsp_slac
from .structure_f import f1p_slac, g1p_slac, g2p_slac

__all__ = ['a1p', 'a2p', 'alp', 'atp']

_alpha = constants.alpha
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3


def a1p_slac(x, q2):
    gamma2 = 4 * _m_p**2 * x**2 / q2
    return (g1p_slac(x, q2) - gamma2 * g2p_slac(x, q2)) / f1p_slac(x, q2)


def a2p_slac(x, q2):
    gamma = numpy.sqrt(4 * _m_p**2 * x**2 / q2)
    return gamma * (g1p_slac(x, q2) + g2p_slac(x, q2)) / f1p_slac(x, q2)


def alp_slac(e, x, q2):
    return dxslp_slac(e, x, q2) / (2 * xsp_slac(e, x, q2))


def atp_slac(e, x, q2):
    return dxstp_slac(e, x, q2) / (2 * xsp_slac(e, x, q2))


def a1p(x, q2, model='slac', **kwargs):
    a1p_func = {
        'slac': a1p_slac,
    }.get(model, None)

    return a1p_func(x, q2, **kwargs)


def a2p(x, q2, model='slac', **kwargs):
    a2p_func = {
        'slac': a2p_slac,
    }.get(model, None)

    return a2p_func(x, q2, **kwargs)


def alp(e, x, q2, model='slac', **kwargs):
    alp_func = {
        'slac': alp_slac,
    }.get(model, None)

    return alp_func(e, x, q2, **kwargs)


def atp(e, x, q2, model='slac', **kwargs):
    atp_func = {
        'slac': atp_slac,
    }.get(model, None)

    return atp_func(e, x, q2, **kwargs)
