#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants

from .structure_f import f1p_slac, f2p_slac, g1p_slac, g2p_slac

__all__ = ['dxslp', 'dxstp', 'xsp']

_alpha = constants.alpha
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6
_inv_gev_to_fm = _inv_fm_to_gev
_inv_gev_to_mkb = _inv_gev_to_fm**2 * 1e4


def _mott(e, theta):
    # mott xs
    cos_theta_2 = numpy.cos(theta / 2)
    sin2_theta_2 = 1 - cos_theta_2**2
    return (_alpha * cos_theta_2 / (2 * e * sin2_theta_2))**2 * _inv_gev_to_mkb


def xsp_slac(e, x, q2):
    # total xs
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))

    return _mott(e, theta) * (2 / _m_p * f1p_slac(x, q2) * numpy.tan(theta / 2)**2 + 1 / nu * f2p_slac(x, q2))


def dxslp_slac(e, x, q2):
    # longitudinal xs difference
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    return sigma0 * ((e + ep * numpy.cos(theta)) * g1p_slac(x, q2) - 2 * _m_p * x * g2p_slac(x, q2))


def dxstp_slac(e, x, q2):
    # transverse xs difference
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    return sigma0 * ep * numpy.sin(theta) * (g1p_slac(x, q2) + 2 * e * g2p_slac(x, q2) / nu)


def xsp(e, x, q2, *, model='slac', **kwargs):
    xsp_func = {
        'slac': xsp_slac,
    }.get(model, None)

    return xsp_func(e, x, q2, **kwargs)


def dxslp(e, x, q2, *, model='slac', **kwargs):
    dxslp_func = {
        'slac': dxslp_slac,
    }.get(model, None)

    return dxslp_func(e, x, q2, **kwargs)


def dxstp(e, x, q2, *, model='slac', **kwargs):
    dxstp_func = {
        'slac': dxstp_slac,
    }.get(model, None)

    return dxstp_func(e, x, q2, **kwargs)
