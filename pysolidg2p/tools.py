#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants, integrate

__all__ = ['dxs_to_g1g2', 'g1g2_to_d2']

_alpha = constants.alpha
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6
_inv_gev_to_fm = _inv_fm_to_gev
_inv_gev_to_mkb = _inv_gev_to_fm**2 * 1e4


def dxs_to_g1g2(e, x, q2, dxsl, dxst, edxsl, edxst):
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    A1 = (e + ep * numpy.cos(theta))
    B1 = -2 * _m_p * x
    C1 = dxsl / sigma0
    A2 = ep * numpy.sin(theta)
    B2 = 2 * e * ep * numpy.sin(theta) / nu
    C2 = dxst / sigma0
    D = A1 * B2 - A2 * B1

    g1 = (C1 * B2 - C2 * B1) / D
    g2 = (-C1 * A2 + C2 * A1) / D

    eC1 = edxsl / sigma0
    eC2 = edxst / sigma0
    eg1 = numpy.sqrt((eC1 * B2 / D)**2 + (eC2 * B1 / D)**2)
    eg2 = numpy.sqrt((eC1 * A2 / D)**2 + (eC2 * A1 / D)**2)

    return g1, g2, eg1, eg2


def g1g2_to_d2(x, g1, g2, eg1, eg2):
    pass
