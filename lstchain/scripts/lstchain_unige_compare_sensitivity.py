#!/usr/bin/env python

import os
import argparse
from lstchain.io.io import read_dl2_to_pyirf, dl2_params_lstcam_key
import logging
import operator
from lstchain.mc import plot_utils
import ctaplot
from ctaplot.plots import plot_angular_resolution_per_energy

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import QTable
import pandas as pd

from pyirf.io.eventdisplay import read_eventdisplay_fits
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM
)
from pyirf.cut_optimization import optimize_gh_cut

from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)

from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)

parser = argparse.ArgumentParser(description="Sensitivity comparison")

# Required argument
parser.add_argument('--input_files', nargs='+', type=str,
                    dest='input_files', required=True,
                    help='Path to the h5 file containing the sensitivity computed data')

parser.add_argument('--labels', nargs='+', type=str,
                    dest='labels', required=True,
                    help='label of the computed data')

parser.add_argument('--output_dir', type=str,
                    dest='output_dir', required=True,
                    help='Path to the directory where you store the final plots')

# Optional arguments
parser.add_argument('--colors', nargs='+', type=str,
                    dest='colors', required=False,
                    help='color for each data')

parser.add_argument('--markers', nargs='+', type=str,
                    dest='markers', required=False,
                    help='markers for each data')


args = parser.parse_args()


# Aleksic et al., 2015, JHEAP, (https://doi.org/10.1016/j.jheap.2015.01.002), fit od 0.1-20 TeV
def crab_magic2015_dNdE(energy):
    spectrum = 3.23 * 10**-11 * energy**(-2.47 - 0.24 * np.log10(energy))   # TeV-1 cm-2 s-1
    #spectrum = spectrum / 1.602176487   # erg-1 cm-2 s-1
    # spectrum = spectrum * 10000 # TeV-1 m-2 s-1
    return spectrum


def plot_magic_sensitivity(ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    magic_table = ctaplot.get_magic_sensitivity()
    magic_table['e_err_lo'] = magic_table['e_center'] - magic_table['e_min']
    magic_table['e_err_hi'] = magic_table['e_max'] - magic_table['e_center']

    key = 'lima_5off'

    if 'ls' not in kwargs and 'linestyle' not in kwargs:
        kwargs['ls'] = ''
    kwargs.setdefault('label', f'MAGIC (Aleksić et al, 2016)')

    k = 'sensitivity_' + key
    ax.errorbar(
        magic_table['e_center'].to_value(u.TeV),
        y=(magic_table['e_center'] ** 2 * magic_table[k]).to_value(u.Unit('TeV cm-2 s-1')),
        xerr=[magic_table['e_err_lo'].to_value(u.TeV), magic_table['e_err_hi'].to_value(u.TeV)],
        yerr=(magic_table['e_center'] ** 2 * magic_table[f'{k}_err']).to_value(u.Unit('TeV cm-2 s-1')),
        **kwargs)

    e_smooth = np.logspace(np.log10(0.001), np.log10(100), 100)
    ax.plot(e_smooth, e_smooth ** 2 * crab_magic2015_dNdE(e_smooth), '-', alpha=0.5, color='grey',
            label='100% Crab')  # 100 %
    ax.plot(e_smooth, e_smooth ** 2 * 0.1 * crab_magic2015_dNdE(e_smooth), '--', alpha=0.5, color='grey',
            label='10% Crab')  # 10 %
    ax.plot(e_smooth, e_smooth ** 2 * 0.01 * crab_magic2015_dNdE(e_smooth), '-.', alpha=0.5, color='grey',
            label='1% Crab')  # 1 %

    ax.set_xlabel(r"$E_{R}$ [TeV]")
    ax.set_ylabel(r"$E^2$ Flux Sensitivity [TeV $cm^{-2} s^{-1}$]")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-13, 1e-9)
    ax.grid('both')

    return ax


def plot_sensitivity(sensitivity_tables, labels, colors, markers, output_dir):
    """
    Makes sensitivity plot of a list of sensitivity tables
    :param sensitivity_tables:      a list of sensitivity tables
    :param labels:                  labels of the list of data
    :param colors:                  color of the list of data
    :param markers:                 markers of the list of data
    :param output_dir:              output directory where to store the plot
    :return:
        nothing, it just save the plot
    """

    fig, ax = plt.subplots()
    plot_magic_sensitivity(ax=ax, color='tab:gray')
    for i, label in enumerate(labels):
        xx = sensitivity_tables[i]['reco_energy_center'].to_value()
        yy = xx ** 2 * sensitivity_tables[i]['flux_sensitivity'].to_value()
        xx_err = 0.5 * (sensitivity_tables[i]['reco_energy_high'] - sensitivity_tables[i]['reco_energy_low']).to_value()
        yy_err = xx ** 2 * [sensitivity_tables[i]['flux_sensitivity_lower_err'].to_value(),
                            sensitivity_tables[i]['flux_sensitivity_upper_err'].to_value()]
        ax.errorbar(x=xx, y=yy, xerr=xx_err, yerr=yy_err, label=labels[i], ecolor=colors[i], color=colors[i], fmt=markers[i])

    ax.legend(ncol=1, loc=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flux_sensitivity.png'), dpi=800)
    plt.close(fig)


def plot_angular_resolution(gamma_tables, proton_tables, labels, colors, markers, output_dir):
    """
    Makes the angular resolution plot of list of tables
    :param gamma_tables:
    :param proton_tables:
    :param labels:                  labels of the list of data
    :param colors:                  color of the list of data
    :param markers:                 markers of the list of data
    :param output_dir:              output directory where to store the plot
    :return:
        nothing, it just save the plot
    """

    fig, ax = plt.subplots()

    for i, gamma_table in enumerate(gamma_tables):

        # Events that passed out the gammaness cut (mix of protons and gammas)
        events = table.vstack(gamma_tables[i][gamma_tables[i]["selected_gh"]],
                              proton_tables[i][proton_tables[i]["selected_gh"]])

        plot_angular_resolution_per_energy(reco_alt=events['reco_alt'].to_value(),
                                           reco_az=events['reco_az'].to_value(),
                                           true_alt=events['true_alt'].to_value(),
                                           true_az=events['true_az'].to_value(),
                                           reco_energy=events['reco_energy'].to_value(),
                                           percentile=68.27,
                                           confidence_level=0.95,
                                           bias_correction=False,
                                           ax=ax,
                                           color=colors[i],
                                           label=labels[i],
                                           marker=markers[i])

    ax.set_xlim(1e-2, 1e2)
    ax.grid('on', which="both")
    ax.set_title(None)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angular_resolution.png'), dpi=800)
    plt.close(fig)


def plot_theta2_distribution(gamma_tables, proton_tables, labels, colors, output_dir):
    """
    Makes the angular resolution plot of list of tables
    :param gamma_tables:
    :param proton_tables:
    :param labels:                  labels of the list of data
    :param colors:                  color of the list of data
    :param markers:                 markers of the list of data
    :param output_dir:              output directory where to store the plot
    :return:
        nothing, it just save the plot
    """

    fig, ax = plt.subplots()

    for i, gamma_table in enumerate(gamma_tables):

        # Events that passed out the gammaness cut (mix of protons and gammas)
        events = table.vstack(gamma_tables[i][gamma_tables[i]["selected_gh"]],
                              proton_tables[i][proton_tables[i]["selected_gh"]])

        ctaplot.plot_theta2(reco_alt=events['reco_alt'].to_value(),
                            reco_az=events['reco_az'].to_value(),
                            true_alt=events['true_alt'].to_value(),
                            true_az=events['true_az'].to_value(),
                            bias_correction=False,
                            ax=ax,
                            color=colors[i],
                            label=labels[i],
                            histtype='step')

    ax.set_title('')
    ax.grid('on', which="both")
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theta2.png'), dpi=800)
    plt.close(fig)


def plot_energy_resolution(gamma_tables, labels, colors, markers, output_dir):
    """
    Makes the energy resolution plot of list of tables
    :param gamma_tables:
    :param labels:                  labels of the list of data
    :param colors:                  color of the list of data
    :param markers:                 markers of the list of data
    :param output_dir:              output directory where to store the plot
    :return:
        nothing, it just save the plot
    """

    fig, ax = plt.subplots()

    for i, gamma_table in enumerate(gamma_tables):

        ctaplot.plot_energy_resolution(true_energy=gamma_table['true_energy'].to_value(),
                                       reco_energy=gamma_table['reco_energy'].to_value(),
                                       ax=ax,
                                       bias_correction=False,
                                       color=colors[i],
                                       marker=markers[i],
                                       label=labels[i])

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_resolution.png'), dpi=800)
    plt.close(fig)


def plot_energy_bias(gamma_tables, labels, colors, markers, output_dir):
    """
    Makes the energy bias plot of list of tables
    :param gamma_tables:
    :param labels:                  labels of the list of data
    :param colors:                  color of the list of data
    :param markers:                 markers of the list of data
    :param output_dir:              output directory where to store the plot
    :return:
        nothing, it just save the plot
    """

    fig, ax = plt.subplots()

    for i, gamma_table in enumerate(gamma_tables):

        ctaplot.plot_energy_bias(true_energy=gamma_table['true_energy'].to_value(),
                                 reco_energy=gamma_table['reco_energy'].to_value(),
                                 ax=ax,
                                 color=colors[i],
                                 marker=markers[i],
                                 label=labels[i])

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_bias.png'), dpi=800)
    plt.close(fig)


def main():

    input_files = args.input_files
    labels = args.labels
    output_dir = args.output_dir

    if args.markers is None:
        markers = ["None"] * len(input_files)
    else:
        markers = args.markers

    if args.colors is None:
        colors = ["None"] * len(input_files)
    else:
        colors = args.colors

    try:
        os.makedirs(output_dir, exist_ok=True)
        print("Directory ", output_dir, " Created ")
    except FileExistsError:
        print("Directory ", output_dir, " already exists")

    sensitivity = []
    gamma_histo = []
    proton_histo = []
    df_gamma = []
    df_proton = []

    if len(input_files) != len(labels):
        print('There must be a label for each input')
        exit()

    for i, file in enumerate(input_files):
        sensitivity.append(QTable.read(file, path='sensitivity/standard'))
        gamma_histo.append(QTable.read(file, path='histogram/gamma_on'))
        proton_histo.append(QTable.read(file, path='histogram/proton'))
        df_gamma.append(QTable.read(file, path='events/gamma_on'))
        df_proton.append(QTable.read(file, path='events/proton'))

    # Plot Flux Sensitivity
    plot_sensitivity(sensitivity_tables=sensitivity,
                     labels=labels,
                     colors=colors,
                     markers=markers,
                     output_dir=output_dir)

    # Angular Resolution
    plot_angular_resolution(gamma_tables=df_gamma,
                            proton_tables=df_proton,
                            labels=labels,
                            colors=colors,
                            markers=markers,
                            output_dir=output_dir)

    # theta2 distribution
    plot_theta2_distribution(gamma_tables=df_gamma,
                             proton_tables=df_proton,
                             labels=labels,
                             colors=colors,
                             output_dir=output_dir)

    # Plot Energy resolution
    plot_energy_resolution(gamma_tables=df_gamma,
                           labels=labels,
                           colors=colors,
                           markers=markers,
                           output_dir=output_dir)

    # Plot Energy Bias
    plot_energy_bias(gamma_tables=df_gamma,
                     labels=labels,
                     colors=colors,
                     markers=markers,
                     output_dir=output_dir)

    # Plot migration matrix
    for i, gamma_table in enumerate(df_gamma):
        # Events that passed out the gammaness cut (mix of protons and gammas)
        events = table.vstack(df_gamma[i][df_gamma[i]["selected_gh"]],
                              df_proton[i][df_proton[i]["selected_gh"]])

        fig, ax = plt.subplots()
        ctaplot.plot_migration_matrix(x=events['log_mc_energy'],
                                      y=events['log_reco_energy'],
                                      ax=ax,
                                      colorbar=True,
                                      xy_line=True,
                                      hist2d_args=dict(norm=mpl.colors.LogNorm()),
                                      line_args=dict(color='black')
                                      )

        ax.set_xlabel(r"$log_{10} E_{MC}$ [TeV]")
        ax.set_ylabel(r"$log_{10} E_{R}$ [TeV]")

        fill_name = labels[i].replace(' ', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'migration_{}.png'.format(fill_name)), dpi=800)
        plt.close(fig)

    # Plot dispersion of the source
        fig, ax = plt.subplots()
        ctaplot.plot_dispersion(events['log_mc_energy'], events['log_reco_energy'], x_log=False, ax=ax)
        fill_name = labels[i].replace(' ', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dispersion_eneergy_{}.png'.format(fill_name)), dpi=800)
        plt.close(fig)

    # Plot rates

# def plot_rate(particle)

# plot_rate(e_min, e_max, rate, rate_err=None, ax=None, **kwargs)

    # Plot effective area

    # Plot Signal to Noise ratio

    # Gammaness and theta cuts


if __name__ == '__main__':
    main()

