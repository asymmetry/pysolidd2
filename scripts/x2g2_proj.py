#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import pickle

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
from scipy import constants

from pysolidg2p import asymmetry, cross_section, sim_reader, structure_f, tools

_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3

#
# input parameters
#

e = 11
x_binning = {'bins': 20, 'range': (0, 1)}
q2_binning = {'bins': 50, 'range': (0.9, 10.9)}
yield_limit = 10000
beam_pol = 0.9
target_pol = 0.7
dilution_factor = 0.13

#
# tool functions
#


def create_bins(binning):
    bins = binning['bins']
    bin_low, bin_high = binning['range']
    bin_centers = numpy.linspace(bin_low + (bin_high - bin_low) / bins / 2, bin_high - (bin_high - bin_low) / bins / 2, bins)
    bin_edges = numpy.linspace(bin_low, bin_high, bins + 1)

    return bin_centers, bin_edges


#
# main program
#

q2_list, q2_edges = create_bins(q2_binning)

x_raw, q2_raw, yield_raw = sim_reader.load('yield.txt')

result = {}

for i, _ in enumerate(q2_list):
    x, x_edges = create_bins(x_binning)
    yield_ = numpy.zeros_like(x)

    for j, _ in enumerate(x):
        q2_x = q2_raw[(x_raw >= x_edges[j]) & (x_raw < x_edges[j + 1])]
        yield_x = yield_raw[(x_raw >= x_edges[j]) & (x_raw < x_edges[j + 1])]

        yield_[j] = numpy.sum(yield_x[(q2_x >= q2_edges[i]) & (q2_x < q2_edges[i + 1])])

    q2 = q2_list[i]
    ep_min = q2 / (4 * 11)
    x_min = q2 / (2 * _m_p * (e - ep_min))
    select = (x > x_min) & (yield_ > yield_limit)
    x = x[select]
    yield_ = yield_[select] * 1000

    asym = asymmetry.atp(e, x, q2)
    easym = 1 / numpy.sqrt(yield_) / (beam_pol * target_pol * dilution_factor)

    xs0 = cross_section.xsp(e, x, q2)
    exs0 = xs0 * (1 / numpy.sqrt(yield_))

    dxsT = 2 * xs0 * asym
    edxsT = dxsT * numpy.sqrt((easym / asym)**2 + (exs0 / xs0)**2)

    dxsL = cross_section.dxslp(e, x, q2)  # from model
    edxsL = 0

    g1, g2, eg1, eg2 = tools.dxs_to_g1g2(e, x, q2, dxsL, dxsT, edxsL, edxsT)

    sub_result = {}
    sub_result['x'] = x
    sub_result['q2'] = q2
    sub_result['g2'] = g2
    sub_result['eg2'] = eg2
    result[str(q2)] = sub_result

#
# output
#

with open('g2.pkl', 'wb') as f:
    pickle.dump(result, f)

x_model = numpy.linspace(0, 1, 201)

with PdfPages('x2g2_proj.pdf') as pdf:
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.axhline(y=0, linestyle='--', linewidth=0.75, color='k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$x^2g_2$')
    plt.xlim(0, 1)

    index_list = [5, 15, 25, 35, 45]
    color_list = ['k', 'r', 'b', 'g', 'c']
    markers = []
    texts = []
    for color, index in zip(color_list, index_list):
        l1, = plt.plot(x_model, x_model**2 * structure_f.g2p(x_model, result[str(q2_list[index])]['q2']), '{}-'.format(color), linewidth=0.75)
        x = result[str(q2_list[index])]['x']
        g2 = result[str(q2_list[index])]['g2']
        eg2 = result[str(q2_list[index])]['eg2']
        p1 = plt.errorbar(x, x**2 * g2, x**2 * eg2, fmt='{}.'.format(color))
        markers.append((p1, l1))
        texts.append(r'${}<Q^2<{}$'.format(q2_edges[index], q2_edges[index + 1]))

    plt.legend(markers, texts)
    pdf.savefig(bbox_inches='tight')
    plt.close()
