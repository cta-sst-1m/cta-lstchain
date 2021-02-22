#!/usr/bin/env python

import os
import argparse
from lstchain.io.io import read_dl2_to_pyirf, dl2_params_lstcam_key
import logging
import operator
from lstchain.mc import plot_utils
import ctaplot

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits
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

log = logging.getLogger("pyirf")
irf = ctaplot.irf_cta()

parser = argparse.ArgumentParser(description="Standard sensitivity computation")

# Required argument
parser.add_argument('--dl2_gamma_test', type=str,
                    dest='dl2_gamma_test',
                    help='Path to the dl2 file of gamma events for testing')

parser.add_argument('--dl2_proton_test', type=str,
                    dest='dl2_proton_test',
                    help='Path to the dl2 file of proton events for testing')

parser.add_argument('--output_dir', type=str,
                    dest='output_dir',
                    help='Output directory to store the data and plots')

parser.add_argument('--alpha', type=float, dest='alpha', default=0.2,
                    help='scaling between ON and OFF region.'
                         'Make the OFF region 5 times larger than the ON region for better background statistics')

parser.add_argument('--observation_time', type=float, dest='observation_time', default=50,
                    help='observation time for compuating the sensitivity, should be in hours')

parser.add_argument('--max_bkg_radius', type=float, dest='max_bkg_radius', default=1.0,
                    help='Radius used in the background rate computation')

parser.add_argument('--initial_gh_cut_efficiency', type=float, dest='initial_gh_cut_efficiency', default=0.5,
                    help='Initial gammaness efficiency used for the first computation of the binned theta cuts. '
                         'IMPORTANT : the initial theta cuts are computed using a fixe g/h cut corresponding to this '
                         'value, then the g/h cuts are optimized once more after this initial theta cuts was applied.')

parser.add_argument('--max_gh_cut_efficiency', type=float, dest='max_gh_cut_efficiency', default=0.8,
                    help='Maximum gammaness efficiency bound for the scans')

parser.add_argument('--gammaness_step', type=float, dest='gammaness_step', default=0.01,
                    help='Steps for the gammaness scan related to the g/h separation efficiency optimization')

parser.add_argument('--nbins', type=int, dest='nbins', default=20,
                    help='Number of energy bins in the chosen energy range. Following CTA standard rules for '
                         'performance evaluation, the energy range is divided in five-per-decade equal logarithmic '
                         'energy bins')

parser.add_argument('--e_min', type=float, dest='e_min', default=0.01,
                    help='Lower energy bound of the chosen energy range in TeV')

parser.add_argument('--e_max', type=float, dest='e_max', default=100,
                    help='Upper energy bound of the chosen energy range in TeV')

parser.add_argument('--intensity_cut', type=float, dest='intensity_cut', default=0,
                    help='Intensity lower cut used in the training to insure consistency with the reconstruction and'
                         'sensitivity computation. By default the lowest value is assumed')

parser.add_argument('--leakage_cut', type=float, dest='leakage_cut', default=1,
                    help='Intensity lower cut used in the training to insure consistency with the reconstruction and'
                         'sensitivity computation. By default the highest values is assumed')

parser.add_argument('--method', type=str, dest='method', default='standard',
                    help='Type of cut optimization method : standard, joint, ect (others to be defined)')

parser.add_argument('--magic_reference', type=str, dest='magic_reference', default=None,
                    help='Path to the MAGIC reference sensitivity')

args = parser.parse_args()


def get_obstime_real(events):
    """
    Calculate the effective observation time
    :param events:         events table
    :return: effective observation time
    """

    deltat = np.diff(events.dragon_time)
    rate = 1/np.mean(deltat[(deltat > 0) & (deltat < 0.1)])
    dead_time = np.amin(deltat)
    t_elapsed = events.shape[0]/rate * u.s
    total_dead_time = events.shape[0]*dead_time
    t_eff = t_elapsed/(1+rate*dead_time)
    print("ELAPSED TIME: %.2f s\n" % t_elapsed.to_value(),
          "EFFECTIVE TIME: %.2f s\n" % t_eff.to_value(),
          "DEAD TIME: %0.2E s\n" % dead_time,
          "TOTAL DEAD TIME: %.2f s\n" % total_dead_time,
          "RATE: %.2f 1/s\n" % rate
          )

    return t_eff


def read_real_dl2_to_pyirf(filename):
    """
    Read DL2 files and convert them into pyirf internal format
    :param filename:       path to the file
    :return: astropy.table.QTable`, `pyirf.simulations.SimulatedEventsInfo`
    """

    # mapping
    name_mapping = {'alt_tel': 'pointing_alt',
                    'az_tel': 'pointing_az',
                    'gammaness': 'gh_score'}

    unit_mapping = {'reco_energy': u.TeV,
                    'pointing_alt': u.rad,
                    'pointing_az': u.rad,
                    'reco_alt': u.rad,
                    'reco_az': u.rad}

    events = pd.read_hdf(filename, key=dl2_params_lstcam_key).rename(columns=name_mapping)
    # Calculate effective observation time
    obstime_real = get_obstime_real(events)

    events = table.QTable.from_pandas(events)

    for k, v in unit_mapping.items():
        events[k] *= v

    return events, obstime_real


def apply_quality_cuts(dataset, intensity_cut, leakage_cut):
    """
    It applies the minimum intensity and max leakage cuts adding an extra column with boolean values for passing
    or not the given quality cut
    :param dataset:
    :param intensity_cut:
    :param leakage_cut:
    :return: same data set with additional quality cuts information
    """

    dataset['good_events'] = (dataset['intensity'] >= intensity_cut) & (dataset['leakage_intensity_width_2'] <= leakage_cut)

    return dataset


def manage_particle(particle_dict, observation_time):
    """
    Modifies a particle dictionary adding extra information
    :param particle_dict:       particle dictionary, e.g. particle['gamma'] or particle['proton']
    :param observation_time:    observational time on source
    :return:
        Nothing, it adds to the input particle dictionary extra information about the particle dataset
    """

    # Getting the simulated spectrum
    particle_dict["simulated_spectrum"] = PowerLaw.from_simulation(particle_dict["simulation_info"], observation_time)
    # Re-weight to target spectrum (for gammas : Crab Hegra)
    particle_dict["events"]["weight"] = calculate_event_weights(particle_dict["events"]["true_energy"],
                                                                particle_dict["target_spectrum"],
                                                                particle_dict["simulated_spectrum"]
                                                                )
    for prefix in ('true', 'reco'):
        k = f"{prefix}_source_fov_offset"
        particle_dict["events"][k] = calculate_source_fov_offset(particle_dict["events"], prefix=prefix)

    particle_dict["events"]["source_fov_offset"] = calculate_source_fov_offset(particle_dict["events"])

    # Compute the theta / distance between the reconstructed and assumed source position
    # only ON observations are handled at this level, i. e. the assumed source position is the pointing position
    particle_dict["events"]["theta"] = calculate_theta(particle_dict["events"],
                                                       assumed_source_az=particle_dict["events"]["true_az"],
                                                       assumed_source_alt=particle_dict["events"]["true_alt"]
                                                       )


def get_initial_theta_cuts(signal, initial_gh_cut_efficiency, energy_bins):
    """
    get the initial thetas cuts for a pre-selection based on a fixe gammaness cut value
    :param signal:                      the gamma data set in QTable format
    :param initial_gh_cut_efficiency:   the initial brute gammaness cut
    :param energy_bins:                 the energy bins for overall sensitivity computation
    :return: The signal data set table with the pre-selection theta cuts called 'selected_theta'
    """

    # Computing pre-selection cut based on an initial gammaness efficiency value
    pre_selection_gh_cut = np.quantile(signal['gh_score'], (1 - initial_gh_cut_efficiency))
    print('Computed pre-selection $\gamma$-ness cut : ', pre_selection_gh_cut)
    print('Base on the initial $\gamma$-ness efficiency cut of  {}'.format(initial_gh_cut_efficiency))

    # Initial theta cut is defined as the 68 percent containment of gammas in each energy bin
    # and in this case, for a fixed global and unoptimized gh (or gammaness) score cut
    mask_theta_cuts = signal["gh_score"] >= pre_selection_gh_cut
    theta_cuts = calculate_percentile_cut(values=signal["theta"][mask_theta_cuts],
                                          bin_values=signal["reco_energy"][mask_theta_cuts],
                                          bins=energy_bins,
                                          fill_value=np.nan * u.deg,
                                          percentile=68)

    # Evaluate the theta pre-selection
    signal["selected_theta"] = evaluate_binned_cut(values=signal["theta"],
                                                   bin_values=signal["reco_energy"],
                                                   cut_table=theta_cuts,
                                                   op=operator.le)

    return signal, theta_cuts


def optimize_gammaness_cuts(signal, background, scan_step, max_scan_val, energy_bins, theta_cuts, alpha, max_bkg_radius):
    """
    Run block below for G/H cut optimization based on best sensitivity
    :param signal:
    :param background:
    :param scan_step:
    :param max_scan_val:
    :param energy_bins:
    :param theta_cuts:
    :param alpha:
    :param max_bkg_radius:
    :return: the signal and background datasets with additional G/H cut boolean table, plus the G/H cuts
    """
    log.info("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(scan_step, max_scan_val + 0.5 * scan_step, scan_step)

    sensitivity, gh_cuts = optimize_gh_cut(signal=signal[signal["selected_theta"]],
                                           background=background,
                                           reco_energy_bins=energy_bins,
                                           gh_cut_efficiencies=gh_cut_efficiencies,
                                           theta_cuts=theta_cuts,
                                           op=operator.ge,
                                           alpha=alpha,
                                           background_radius=max_bkg_radius)

    return signal, background, gh_cuts


def compute_error(sensitivity, signal, background, theta_cuts, max_bkg_radius, alpha):
    """
    Compute errors on the sensitivity based on the number of simulated signal and background selected
    :param sensitivity:         Sensitivity QTable without errors
    :param signal:              Signal histogram defined as a QTable (per energy bin)
    :param background:          Background histogram defined as a QTable (per energy bin)
    :param alpha:               Alpha ratio of observation ON/OFF
    :param theta_cuts:          optimized theta cuts
    :param max_bkg_radius:      maximum background radius to increase statistics
    :return: sensitivity QTable with errors

    """
    for element in (signal, background):

        lower_err = QTable()
        upper_err = QTable()

        for error in (lower_err, upper_err):
            error["reco_energy_low"] = element["reco_energy_low"]
            error["reco_energy_high"] = element["reco_energy_high"]
            error["reco_energy_center"] = element["reco_energy_center"]

        weights = element["n_weighted"].value / element["n"]
        if element is signal:
            errn = np.sqrt(element["n"])
        else:
            errn = np.sqrt(element["n"] / alpha) * theta_cuts["cut"].value / max_bkg_radius.value

        errn_weighted = errn * weights
        lower_err["n"] = element["n"] - errn
        lower_err["n_weighted"] = element["n_weighted"] - errn_weighted
        upper_err["n"] = element["n"] + errn
        upper_err["n_weighted"] = element["n_weighted"] + errn_weighted

        upper_err["n_weighted"][upper_err["n_weighted"] < 0] = 0

        if element is signal:
            signal_lower_err = lower_err
            signal_upper_err = upper_err
        else:
            background_lower_err = lower_err
            background_upper_err = upper_err

    sensitivity_lower_err = calculate_sensitivity(signal_hist=signal_upper_err,
                                                  background_hist=background_lower_err,
                                                  alpha=alpha)

    sensitivity_upper_err = calculate_sensitivity(signal_hist=signal_lower_err,
                                                  background_hist=background_upper_err,
                                                  alpha=alpha)

    sensitivity["relative_sensitivity_lower_err"] = (
                sensitivity_lower_err["relative_sensitivity"] - sensitivity["relative_sensitivity"])
    sensitivity["relative_sensitivity_upper_err"] = (
                sensitivity["relative_sensitivity"] - sensitivity_upper_err["relative_sensitivity"])

    return sensitivity


def main():

    dl2_gamma_test = args.dl2_gamma_test
    dl2_proton_test = args.dl2_proton_test
    output_dir = args.output_dir

    alpha = args.alpha
    observation_time = args.observation_time * u.hour
    max_bkg_radius = args.max_bkg_radius * u.deg
    initial_gh_cut_efficiency = args.initial_gh_cut_efficiency
    max_gh_cut_efficiency = args.max_gh_cut_efficiency
    gammaness_step = args.gammaness_step
    nbins = args.nbins
    e_min = args.e_min
    e_max = args.e_max
    intensity_cut = args.intensity_cut
    leakage_cut = args.leakage_cut
    method = args.method

    magic_reference = args.magic_reference

    output_plot = os.path.join(output_dir, 'plot')
    output_data = os.path.join(output_dir, 'data')

    try:
        os.makedirs(output_plot, exist_ok=True)
        print("Directory ", output_data, " Created ")
        os.makedirs(output_data, exist_ok=True)
        print("Directory ", output_plot, " Created ")
    except FileExistsError:
        print("Directory ", output_plot, " already exists")
        print("Directory ", output_data, " already exists")

    gamma_events, gamma_simu_info = read_dl2_to_pyirf(dl2_gamma_test)
    proton_events, proton_simu_info = read_dl2_to_pyirf(dl2_proton_test)

    print('simulation info for gammas : ')
    print(gamma_simu_info)
    print('simulation info for protons : ')
    print(proton_simu_info)

    gamma_events = apply_quality_cuts(dataset=gamma_events, intensity_cut=intensity_cut, leakage_cut=leakage_cut)
    proton_events = apply_quality_cuts(dataset=proton_events, intensity_cut=intensity_cut, leakage_cut=leakage_cut)

    # Definition of a particle dictionary for easy data access
    particles = {
        "gamma": {
            "events": gamma_events[gamma_events['good_events']],
            "simulation_info": gamma_simu_info,
            "target_spectrum": CRAB_HEGRA
        },
        "proton": {
            "events": proton_events[proton_events['good_events']],
            "simulation_info": proton_simu_info,
            "target_spectrum": IRFDOC_PROTON_SPECTRUM
        }
    }

    manage_particle(particles['gamma'], observation_time)
    manage_particle(particles['proton'], observation_time)

    # Selecting only the events
    gammas = particles["gamma"]["events"]
    protons = particles["proton"]["events"]

    # Data to optimize best cuts
    signal = gammas
    background = protons

    energy_bins = np.logspace(np.log10(e_min), np.log10(e_max), nbins + 1) * u.TeV

    # Compute the best cuts for sensitivity
    if method is 'standard':

        signal, theta_cuts = get_initial_theta_cuts(signal=signal,
                                                    initial_gh_cut_efficiency=initial_gh_cut_efficiency,
                                                    energy_bins=energy_bins)

        signal, background, gh_cuts = optimize_gammaness_cuts(signal=signal,
                                                              background=background,
                                                              scan_step=gammaness_step,
                                                              max_scan_val=max_gh_cut_efficiency,
                                                              energy_bins=energy_bins,
                                                              theta_cuts=theta_cuts,
                                                              alpha=alpha,
                                                              max_bkg_radius=max_bkg_radius)

        # Evaluate G/H cut
        for tab in (gammas, protons):
            tab["selected_gh"] = evaluate_binned_cut(values=tab["gh_score"],
                                                     bin_values=tab["reco_energy"],
                                                     cut_table=gh_cuts,
                                                     op=operator.ge)

        # Setting optimal theta cuts after sensitivity optimization according to gammaness
        theta_cuts_opt = calculate_percentile_cut(values=gammas[gammas["selected_gh"]]["theta"],
                                                  bin_values=gammas[gammas["selected_gh"]]["reco_energy"],
                                                  bins=energy_bins,
                                                  percentile=68,
                                                  fill_value=0.32 * u.deg)

        # Applying the optimized cuts into the gamma and proton datasets
        gammas["selected_theta"] = evaluate_binned_cut(values=gammas["theta"],
                                                       bin_values=gammas["reco_energy"],
                                                       cut_table=theta_cuts_opt,
                                                       op=operator.le)

        # Making a selection of the particles satisfying the cuts
        gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]
        protons["selected"] = protons["selected_gh"]

    else:
        print(r'Joint gammaness and $\theta^2$ method not implemented yet')
        exit(1)

    # Create Event Histograms for signal and background
    gamma_hist = create_histogram_table(events=gammas[gammas["selected"]],
                                        bins=energy_bins)

    proton_hist = estimate_background(events=protons[protons["selected"]],
                                      reco_energy_bins=energy_bins,
                                      theta_cuts=theta_cuts_opt,
                                      alpha=alpha,
                                      background_radius=max_bkg_radius)

    sensitivity = calculate_sensitivity(signal_hist=gamma_hist, background_hist=proton_hist, alpha=alpha)
    sensitivity = compute_error(sensitivity=sensitivity,
                                signal=gamma_hist,
                                background=proton_hist,
                                theta_cuts=theta_cuts_opt,
                                max_bkg_radius=max_bkg_radius,
                                alpha=alpha)

    # Adding the cuts to the sensitivity table
    sensitivity["gh_cut"] = gh_cuts["cut"]
    sensitivity["theta_cut"] = theta_cuts_opt["cut"]

    # Scaling relative sensitivity to Crab flux to get the flux sensitivity
    spectrum = particles['gamma']['target_spectrum']
    sensitivity["flux_sensitivity"] = (sensitivity["relative_sensitivity"] * spectrum(sensitivity["reco_energy_center"]))
    sensitivity["flux_sensitivity_lower_err"] = (sensitivity["relative_sensitivity_lower_err"] * spectrum(sensitivity["reco_energy_center"]))
    sensitivity["flux_sensitivity_upper_err"] = (sensitivity["relative_sensitivity_upper_err"] * spectrum(sensitivity["reco_energy_center"])
    )

    # Rates
    area_ratio_p = (1 - np.cos(theta_cuts_opt['cut'])) / (1 - np.cos(max_bkg_radius))
    gamma_hist["rates"] = gamma_hist["n_weighted"] / observation_time.to(u.min)
    proton_hist["rates"] = proton_hist["n_weighted"] * area_ratio_p / observation_time.to(u.min)

    gamma_hist["rates_50h"] = gamma_hist["n_weighted"]
    proton_hist["rates_50h"] = proton_hist["n_weighted"] * area_ratio_p

    # Saving final sensitivity into a file
    output_data_file = os.path.join(output_data, 'sensitivity.hdf5')
    if os.path.exists(output_data_file):
        os.remove(output_data_file)
    else:
        print("Can't delete the file as it doesn't exists")

    sensitivity.write(output_data_file, path='sensitivity/standard', serialize_meta=True)
    gammas.write(output_data_file, path='events/gamma_on', serialize_meta=True, append=True)
    protons.write(output_data_file, path='events/proton', serialize_meta=True, append=True)

    proton_hist.write(output_data_file, path='histogram/proton', serialize_meta=True, append=True)
    gamma_hist.write(output_data_file, path='histogram/gamma_on', serialize_meta=True, append=True)

    print("sensitivity.hdf5 saved as {}".format(output_data_file))

    # Make some default plots :

    # Sensitivity
    fig, ax = plt.subplots(figsize=(12, 8))
    unit = u.Unit('TeV cm-2 s-1')

    energy = sensitivity['reco_energy_center']
    source = (energy ** 2 * sensitivity['flux_sensitivity'])

    ax.errorbar(energy.to_value(u.GeV),
                source.to_value(unit), xerr=(sensitivity['reco_energy_high'] - sensitivity['reco_energy_low']).to_value(u.GeV) / 2,
                label='MC gammas/protons')

    # Get Magic sensitivity
    if magic_reference is not None:
        magic = np.loadtxt(magic_reference, skiprows=1)
        ax.loglog(magic[:, 0], magic[:, 3] * np.power(magic[:, 0] / 1e3, 2), color='tab:gray', label='$MAGIC_{STEREO}$')

    # Get Crab SED
    plot_utils.plot_Crab_SED(ax, 100, 5 * u.GeV, 1e4 * u.GeV, label="100% Crab")  # Energy in GeV
    plot_utils.plot_Crab_SED(ax, 10, 5 * u.GeV, 1e4 * u.GeV, linestyle='--', label="10% Crab")  # Energy in GeV
    plot_utils.plot_Crab_SED(ax, 1, 5 * u.GeV, 1e4 * u.GeV, linestyle=':', label="1% Crab")  # Energy in GeV

    ax.set_title('Minimal Flux Needed for 5σ Detection in 50 hours')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed energy [GeV]")
    ax.set_ylabel(rf"$(E^2 \cdot \mathrm{{Flux Sensitivity}}) /$ ({unit.to_string('latex')})")
    ax.grid(which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'sensitivity.png'), dpi=600)
    plt.close()

    # Rates
    area_ratio_p = (1 - np.cos(theta_cuts_opt['cut'])) / (1 - np.cos(max_bkg_radius))
    rate_gammas = gamma_hist["n_weighted"] / observation_time.to(u.min)
    rate_proton = proton_hist["n_weighted"] * area_ratio_p / observation_time.to(u.min)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.errorbar(0.5 * (gamma_hist['reco_energy_low'] + gamma_hist['reco_energy_high']).to_value(u.TeV),
                rate_gammas.to_value(1 / u.min),
                xerr=0.5 * (gamma_hist['reco_energy_high'] - gamma_hist['reco_energy_low']).to_value(u.TeV),
                label='Gammas MC')
    ax.errorbar(0.5 * (proton_hist['reco_energy_low'] + proton_hist['reco_energy_high']).to_value(u.TeV),
                 rate_proton.to_value(1 / u.min),
                 xerr=0.5 * (proton_hist['reco_energy_high'] - proton_hist['reco_energy_low']).to_value(u.TeV),
                 label='Protons MC')

    ax.legend()
    ax.set_ylabel('Rate events/min')
    ax.set_xlabel(r'$E_{RECO}$ [TeV]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'rates.png'), dpi=600)
    plt.close()

    # Cuts Gammaness and theta
    fig, ax0 = plt.subplots()

    ax0.errorbar(0.5 * (theta_cuts['low'] + theta_cuts['high']).to_value(u.TeV),
                (theta_cuts['cut'] ** 2).to_value(u.deg ** 2),
                xerr=0.5 * (theta_cuts['high'] - theta_cuts['low']).to_value(u.TeV),
                ls='', color='tab:blue')

    ax1 = ax0.twinx()

    ax1.errorbar(0.5 * (gh_cuts['low'] + gh_cuts['high']).to_value(u.TeV),
                 gh_cuts['cut'],
                 xerr=0.5 * (gh_cuts['high'] - gh_cuts['low']).to_value(u.TeV),
                 ls='', color='tab:red')

    ax0.set_ylabel(r'$\theta^2$ cut', color='tab:blue')
    ax0.set_xscale('log')
    ax0.grid('on', which="both")

    ax1.set_ylabel('G/H cut', color='tab:red')
    ax1.set_xlabel(r"$E_{RECO}$ [TeV]")
    ax1.set_xscale('log')
    ax1.grid(which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'cuts.png'), dpi=600)
    plt.close()

    # Angular Resolution
    selected_events_gh = table.vstack(gammas[gammas["selected_gh"]], protons[protons["selected_gh"]])
    ang_res = angular_resolution(selected_events_gh[selected_events_gh["selected_gh"]], energy_bins)

    fig, ax = plt.subplots()
    ax.errorbar(0.5 * (ang_res['true_energy_low'] + ang_res['true_energy_high']),
                ang_res['angular_resolution'],
                xerr=0.5 * (ang_res['true_energy_high'] - ang_res['true_energy_low']),
                ls='')

    ax.set_xlim(1.e-2, 2.e2)
    ax.set_ylim(0.5e-1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_{MC}$ [TeV]")
    ax.set_ylabel("Angular Resolution [deg]")
    ax.grid(which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'angular_resolution.png'), dpi=600)
    plt.close()

    # Energy resolution
    selected_events = table.vstack(gammas[gammas["selected"]], protons[protons["selected"]])
    bias_resolution = energy_bias_resolution(selected_events, energy_bins)

    fig, ax = plt.subplots()
    ax.errorbar(0.5 * (bias_resolution['true_energy_low'] + bias_resolution['true_energy_high']),
                bias_resolution['resolution'],
                xerr=0.5 * (bias_resolution['true_energy_high'] - bias_resolution['true_energy_low']),
                ls='')

    ax.set_xscale('log')
    ax.set_xlabel(r"$E_{MC}$ [TeV]")
    ax.set_ylabel("Energy resolution")
    ax.grid(which="both")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'energy_resolution.png'), dpi=600)
    plt.close()

    # Reco Alt/Az for MC selected events
    emin_bins = [0.0, 0.1, 0.5, 1, 5] * u.TeV
    emax_bins = [0.1, 0.5, 1, 5, 10] * u.TeV

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 4))

    for i, ax in enumerate(axs):
        events = selected_events[(selected_events['reco_energy'] > emin_bins[i]) & \
                                 (selected_events['reco_energy'] < emax_bins[i])]
        pcm = ax.hist2d(events['reco_az'].to_value(u.deg), events['reco_alt'].to_value(u.deg), bins=50)
        ax.title.set_text("%.1f-%.1f TeV" % (emin_bins[i].to_value(), emax_bins[i].to_value()))
        ax.set_xlabel("Az (º)")
        ax.set_ylabel("Alt (º)")
        fig.colorbar(pcm[3], ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'pointing_reco.png'), dpi=600)
    plt.close()

    # Checks on number of islands
    gammas_selected = gammas[(gammas['selected']) & (gammas['reco_energy'] > 1. * u.TeV)]
    protons_selected = protons[(protons['selected']) & (protons['reco_energy'] > 1. * u.TeV)]

    fig, ax = plt.subplots()

    ax.hist(gammas_selected['n_islands'], bins=10, range=(0.5, 10.5), alpha=0.45, color='tab:blue', label='$\gamma$')
    ax.hist(protons_selected['n_islands'], bins=10, range=(0.5, 10.5), alpha=0.45, color='tab:orange', label='p')

    ax.set_yscale('log')
    ax.set_ylabel('Counts')
    ax.set_xlabel('number of island []')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot, 'n_island.png'), dpi=600)
    plt.close()


if __name__ == '__main__':
    main()