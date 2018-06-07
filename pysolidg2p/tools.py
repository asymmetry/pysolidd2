#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants

from .structure_f import r

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


def atf1a1_to_g2(e, x, q2, at, f1, a1, eat, ef1, ea1):
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))

    gamma2 = 4 * _m_p**2 * x**2 / q2
    epsilon = 1 / (1 + 2 * (1 + 1 / gamma2) * numpy.tan(theta / 2)**2)

    rr, err = r(x, q2)

    C = nu * f1 / (2 * e)
    A = C * (at * nu * ((1 + epsilon * rr) / (1 - epsilon)) / (ep * numpy.sin(theta)))
    B = C * a1

    g2 = (A - B) / (1 + gamma2 * nu / (2 * e))
    err_to_eg2 = C * (at * nu * (epsilon * err / (1 - epsilon)) / (ep * numpy.sin(theta))) / (1 + gamma2 * nu / (2 * e))
    eg2 = numpy.sqrt((g2 * ef1 / f1)**2 + (A * eat / at)**2 + (B * ea1 / a1)**2 + (err_to_eg2)**2)

    return g2, eg2


def g1g2_to_d2(x, g1, g2, eg1, eg2):
    dx = (x[-1] - x[0]) / (len(x) - 1)

    integrand = x**2 * (2 * g1 + 3 * g2)
    eintegrand = x**2 * numpy.sqrt(4 * eg1**2 + 9 * eg2**2)

    trapz_integrand = integrand * 2
    trapz_integrand[[0, -1]] = trapz_integrand[[0, -1]] / 2
    trapz = numpy.sum(trapz_integrand) * dx / 2

    trapz_eintegrand = eintegrand * 2
    trapz_eintegrand[[0, -1]] = trapz_eintegrand[[0, -1]] / 2
    etrapz_sum = numpy.sqrt(numpy.sum(trapz_eintegrand**2)) * dx / 2

    etrapz_sys = (x[-1] - x[0])**3 / (12 * (len(x) - 1)**2) * numpy.max(numpy.abs(numpy.gradient(integrand, 2)))

    etrapz = numpy.sqrt(etrapz_sum**2 + etrapz_sys**2)

    return trapz, etrapz
