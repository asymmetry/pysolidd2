#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import pickle

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
from scipy import constants

from pysolidg2p import asymmetry, cross_section, experiments, sim_reader, structure_f, tools

_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3


def get_d2(filename, q2_max):
    with open(filename, 'rb') as f:
        table = pickle.load(f)

    q2_list = []
    d2_list = []
    ed2_list = []
    for _, a_point in table.items():
        x = a_point['x']
        q2 = a_point['q2']
        g2 = a_point['g2']
        eg2 = a_point['eg2']

        if q2 < 1.5 or q2 > q2_max or len(x) < 4:
            continue

        if eg2[0] > 0.01 * numpy.abs(g2[0]):
            x = x[1:]
            g2 = g2[1:]
            eg2 = eg2[1:]

        g1 = structure_f.g1p(x, q2)
        eg1 = g1 * experiments.e12_06_109(x, q2)  # relative error

        x_l = numpy.linspace(1e-3, x[0], 50)
        g1_l = structure_f.g1p(x_l, q2)
        eg1_l = g1_l * 0.1
        g2_l = structure_f.g2p(x_l, q2)
        eg2_l = g2_l * 0.1
        d2_l, ed2_l = tools.g1g2_to_d2(x_l, g1_l, g2_l, eg1_l, eg2_l)

        x_h = numpy.linspace(x[-1], 1, 5)
        g1_h = structure_f.g1p(x_h, q2)
        eg1_h = g1_h * 0.1
        g2_h = structure_f.g2p(x_h, q2)
        eg2_h = g2_h * 0.1
        d2_h, ed2_h = tools.g1g2_to_d2(x_h, g1_h, g2_h, eg1_h, eg2_h)

        d2_0, ed2_0 = tools.g1g2_to_d2(x, g1, g2, eg1, eg2)

        d2 = d2_l + d2_0 + d2_h
        ed2 = numpy.sqrt(ed2_l**2 + ed2_0**2 + ed2_h**2)

        q2_list.append(q2)
        d2_list.append(d2)
        ed2_list.append(ed2)

    return q2_list, d2_list, ed2_list


q2_11gev, d2_11gev, ed2_11gev = get_d2('g2_11gev.pkl', 9.5)
q2_11gev = numpy.array(q2_11gev)
d2_11gev = numpy.array(d2_11gev)
ed2_11gev = numpy.array(ed2_11gev)

q2_8gev, d2_8gev, ed2_8gev = get_d2('g2_8.8gev.pkl', 8.0)
q2_8gev = numpy.array(q2_8gev)
d2_8gev = numpy.array(d2_8gev)
ed2_8gev = numpy.array(ed2_8gev)

q2_slac = numpy.array([5])
d2_slac = numpy.array([0.0032])
ed2_slac = numpy.array([0.0017])

q2_hermes = numpy.array([5-0.1])
d2_hermes = numpy.array([0.0148])
ed2_hermes = numpy.array([0.0107])

q2_rss = numpy.array([1.28])
d2_rss = numpy.array([0.0104])
ed2_rss = numpy.array([0.00136])

q2_lattice = numpy.array([5+0.1])
d2_lattice = numpy.array([0.017])
ed2_lattice = numpy.array([0.007])

with PdfPages('d2_proj.pdf') as pdf:
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    ax.axhline(y=0, linestyle='--', linewidth=0.75, color='k')
    plt.xlabel(r'$Q^2$')
    plt.ylabel(r'$d_2$')
    plt.xlim(1, 10)
    plt.ylim(-0.015, 0.03)
    p1 = plt.errorbar(q2_11gev, d2_11gev, ed2_11gev, fmt='r.', elinewidth=1, capsize=1.5)
    p2 = plt.errorbar(q2_8gev, d2_8gev-0.003, ed2_8gev, fmt='b.', elinewidth=1, capsize=1.5)
    p3 = plt.errorbar(q2_slac, d2_slac, ed2_slac, fmt='ko', elinewidth=1, capsize=2, fillstyle='none')
    p4 = plt.errorbar(q2_hermes, d2_hermes, ed2_hermes, fmt='kd', elinewidth=1, capsize=2, fillstyle='none')
    p5 = plt.errorbar(q2_rss, d2_rss, ed2_rss, fmt='k^', elinewidth=1, capsize=2, fillstyle='none')
    p6 = plt.errorbar(q2_lattice, d2_lattice, ed2_lattice, fmt='ks', elinewidth=1, capsize=2, ms=4)
    plt.legend([(p1), (p2), (p3), (p4), (p5), (p6)], ['11GeV Projection', '8.8GeV Projection', 'SLAC E155', 'HERMES', 'RSS', 'Lattice QCD'])
    pdf.savefig(bbox_inches='tight')
    plt.close()
