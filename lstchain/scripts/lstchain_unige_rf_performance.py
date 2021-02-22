#!/usr/bin/env python

import argparse
import logging
import sys
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lstchain.io import standard_config, replace_config, read_configuration_file
from lstchain.io.io import dl1_params_lstcam_key
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events
from lstchain.visualization import plot_dl2
from lstchain.reco import utils
from lstchain.io import standard_config, replace_config, read_configuration_file
from lstchain.io.io import dl1_params_lstcam_key

import ctaplot
import astropy.units as u
import joblib

irf = ctaplot.irf_cta()


parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--config_file', '-c', type=str,
                    dest='config_file',
                    help='Path to the dl1 file of gamma events for training')

parser.add_argument('--dl1_gamma_train', type=str,
                    dest='dl1_gamma_train',
                    help='Path to the dl1 file of gamma events for training')

parser.add_argument('--dl1_proton_train', type=str,
                    dest='dl1_proton_train',
                    help='Path to the dl1 file of proton events for training')

parser.add_argument('--dl1_gamma_test', type=str,
                    dest='dl1_gamma_test',
                    help='Path to the dl1 file of gamma events for testing')

parser.add_argument('--dl1_proton_test', type=str,
                    dest='dl1_proton_test',
                    help='Path to the dl1 file of proton events for testing')

parser.add_argument('--path_to_models', type=str,
                    dest='path_to_models',
                    help='Path to the trained RF models')

parser.add_argument('--dl1_cam_key', type=str,
                    dest='dl1_cam_key',
                    default='dl1/event/telescope/parameters/LST_LSTCam',
                    help='dl1 cam key')

parser.add_argument('--output_dir', type=str,
                    dest='output_dir',
                    help='Output directory to store the data and plots')


args = parser.parse_args()


def plot_features(data, intensity_cut, leakage_cut, r_cut, output_dir, true_particle=False):

    type_label = 'reco_type'
    if true_particle:
        type_label = 'mc_type'

    # Energy distribution
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['log_mc_energy'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['log_mc_energy'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"$log_{10}E$ [GeV]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_log_mc_energy.png', facecolor='white', dpi=600)
    plt.close(fig)

    # disp distribution
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['disp_norm'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['disp_norm'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"disp_norm [m]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_disp_norm.png', facecolor='white', dpi=600)
    plt.close(fig)

    # Intensity distribution LOG
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['log_intensity'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['log_intensity'], histtype=u'step', bins=100, label="p")
    ax.axvline(np.log10(intensity_cut), linestyle='dashed', color='tab:gray',
               label='cut : {} [pe]'.format(intensity_cut))
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"$log_{10} Intensity$ []")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_log_intensity.png', facecolor='white', dpi=600)
    plt.close(fig)

    # Leakage distribution
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['leakage_intensity_width_2'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['leakage_intensity_width_2'], histtype=u'step', bins=100, label="p")
    ax.axvline(leakage_cut, linestyle='dashed', color='tab:gray', label='cut : {}'.format(leakage_cut))
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"Leakage []")
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_leakage.png', facecolor='white', dpi=600)
    plt.close(fig)

    dataforwl = data[data['log_intensity'] > np.log10(intensity_cut)]

    # Width distribution
    fig, ax = plt.subplots()
    ax.hist(dataforwl[dataforwl[type_label] == 0]['width'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(dataforwl[dataforwl[type_label] == 101]['width'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"Width [º]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_width.png', facecolor='white', dpi=600)
    plt.close(fig)

    # Length distribution
    fig, ax = plt.subplots()
    ax.hist(dataforwl[dataforwl[type_label] == 0]['length'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(dataforwl[dataforwl[type_label] == 101]['length'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"Length [º]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_length.png', facecolor='white', dpi=600)
    plt.close(fig)

    # r distribution
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['r'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['r'], histtype=u'step', bins=100, label="p")
    ax.axvline(r_cut, linestyle='dashed', color='tab:gray', label='cut : {} [m]'.format(r_cut))
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"Radius to c.o.g [m]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_r.png', facecolor='white', dpi=600)
    plt.close(fig)

    # psi distribution
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['psi'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['psi'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"$\Psi$ [rad]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_psi.png', facecolor='white', dpi=600)
    plt.close(fig)

    # phi distribution
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['phi'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['phi'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_phi.png', facecolor='white', dpi=600)
    plt.close(fig)

    # Time gradient
    fig, ax = plt.subplots()
    ax.hist(data[data[type_label] == 0]['time_gradient'], histtype=u'step', bins=100, label="$\gamma$")
    ax.hist(data[data[type_label] == 101]['time_gradient'], histtype=u'step', bins=100, label="p")
    ax.set_ylabel(r'Counts', fontsize=15)
    ax.set_xlabel(r"Time gradient []")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + '/feat_time_gradient.png', facecolor='white', dpi=600)
    plt.close(fig)


def main():

    config_file = args.config_file
    dl1_gamma_train = args.dl1_gamma_train
    dl1_gamma_test = args.dl1_gamma_test
    dl1_proton_train = args.dl1_proton_train
    dl1_proton_test = args.dl1_proton_test
    path_to_models = args.path_to_models
    dl1_cam_key = args.dl1_cam_key
    output_dir = args.output_dir

    custom_config = {}

    if config_file is not None:
        try:
            custom_config = read_configuration_file(config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    reg_energy = joblib.load(path_to_models + '/reg_energy.sav')
    reg_disp_vector = joblib.load(path_to_models + '/reg_disp_vector.sav')
    cls_gh = joblib.load(path_to_models + '/cls_gh.sav')

    gammas = filter_events(pd.read_hdf(dl1_gamma_test, key=dl1_cam_key), config["events_filters"])
    proton = filter_events(pd.read_hdf(dl1_proton_test, key=dl1_cam_key), config["events_filters"])
    data = pd.concat([gammas, proton], ignore_index=True)

    dl2 = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_vector, custom_config=config)

    selected_gammas = dl2.query('reco_type==0 & mc_type==0')

    if (len(selected_gammas) == 0):
        print('No gammas selected, I will not plot any output')
        sys.exit()

    for a_type in ['reco_type', 'mc_type']:

        output_dir_type = os.path.join(output_dir, a_type)

        try:
            os.makedirs(output_dir_type, exist_ok=True)
            print("Directory ", output_dir_type, " Created ")
        except FileExistsError:
            print("Directory ", output_dir_type, " already exists")

        if a_type is 'reco_type':
            true_particle = False
        elif a_type is 'mc_type':
            true_particle = True
        else:
            print('unknown type : mc or reco?')
            exit(1)

        plot_features(data=dl2,
                      intensity_cut=config['events_filters']['intensity'][0],
                      leakage_cut=config['events_filters']['leakage_intensity_width_2'][1],
                      r_cut=config['events_filters']['r'][1],
                      output_dir=output_dir_type,
                      true_particle=true_particle
                      )

    # Compute energy resolution and bias
    if os.path.exists(os.path.join(output_dir, 'e_reso.h5')):
        os.remove(os.path.join(output_dir, 'e_reso.h5'))
    plot_dl2.energy_results(dl2_data=selected_gammas,
                            points_outfile=os.path.join(output_dir, 'e_reso.h5'),
                            plot_outfile=os.path.join(output_dir, 'e_reso_all.png')
                            )

    # Plot energy resolution and bias
    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()
    ctaplot.plot_energy_resolution(selected_gammas.mc_energy,
                                   selected_gammas.reco_energy,
                                   ax=ax0,
                                   bias_correction=False,
                                   color='tab:red',
                                   label='resolution')

    ctaplot.plot_energy_resolution_cta_requirement('north', ax=ax0, color='tab:gray', linestyle='dashed')
    ctaplot.plot_energy_bias(selected_gammas.mc_energy,
                             selected_gammas.reco_energy,
                             ax=ax1,
                             label='Bias')

    ax1.set_ylabel(r"Energy bias", color='tab:blue')
    ax0.set_ylabel(r"${\Delta E/E}_{68}$", color='tab:red')
    ax0.set_xlabel(r"$E_{RECO} [TeV]$")

    ax0.set_title("")
    ax1.set_title("")
    ax0.set_ylim(-0.1, 0.5)
    ax1.set_ylim(-0.3, 0.3)

    ax0.get_legend().remove()
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    h = h0 + h1
    l = l0 + l1
    box = (0.5, 1.02)
    ax0.legend(handles=h, loc="lower center", ncol=3,
               bbox_to_anchor=box, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e_reso_bias.png'), facecolor='white', dpi=800)

    # Plot energy resolution and bias
    fig, ax = plt.subplots()
    ctaplot.plot_migration_matrix(selected_gammas.mc_energy.apply(np.log10),
                                  selected_gammas.reco_energy.apply(np.log10),
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  ax=ax
                                  )

    ax.set_xlabel(r"$log_{10} E_{MC}$ [TeV]")
    ax.set_ylabel(r"$log_{10} E_{RECO}$ [TeV]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e_migration.png'), facecolor='white', dpi=600)

    # Compute angular resolution
    if os.path.exists(os.path.join(output_dir, 'angular_resolution.h5')):
        os.remove(os.path.join(output_dir, 'angular_resolution.h5'))
    plot_dl2.direction_results(selected_gammas,
                               points_outfile=os.path.join(output_dir, 'angular_resolution.h5'),
                               plot_outfile=os.path.join(output_dir, 'angular_resolution_all.png')
                               )

    # Plot angular resolution
    fig, ax = plt.subplots()
    ctaplot.plot_theta2(selected_gammas.reco_alt,
                        selected_gammas.reco_az,
                        selected_gammas.mc_alt,
                        selected_gammas.mc_az,
                        ax=ax,
                        bins=100,
                        range=(0, 1)
                        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theta2.png'), facecolor='white', dpi=600)

    fig, ax = plt.subplots()

    ctaplot.plot_angular_resolution_per_energy(selected_gammas.reco_alt,
                                               selected_gammas.reco_az,
                                               selected_gammas.mc_alt,
                                               selected_gammas.mc_az,
                                               selected_gammas.reco_energy,
                                               ax=ax,
                                               label='resolution'
                                               )

    ctaplot.plot_angular_resolution_cta_requirement('north', ax=ax, color='tab:gray', linestyle='dashed')

    ax.set_title("")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel(r"$E_{RECO} [TeV]$")

    h, l = ax.get_legend_handles_labels()
    box = (0.5, 1.02)
    ax.legend(handles=h, loc="lower center", ncol=3,
              bbox_to_anchor=box, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angular_resolution.png'), facecolor='white', dpi=600)

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ctaplot.plot_migration_matrix(selected_gammas.disp_dx,
                                  selected_gammas.reco_disp_dx,
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax0
                                  )

    ctaplot.plot_migration_matrix(selected_gammas.disp_dy,
                                  selected_gammas.reco_disp_dy,
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax1
                                  )

    ax0.set_xlabel('$DISP_{MC}$')
    ax0.set_ylabel('$DISP_{RECO}$')
    ax0.set_title('DIPS dx')

    ax1.set_xlabel('$DISP_{MC}$')
    ax1.set_title('DISP dy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disp_vector.png'), facecolor='white', dpi=800)

    trueX = dl2[dl2['mc_type'] == 0]['src_x']
    trueY = dl2[dl2['mc_type'] == 0]['src_y']
    recoX = dl2[dl2['reco_type'] == 0]['reco_src_x']
    recoY = dl2[dl2['reco_type'] == 0]['reco_src_y']

    fig, ax = plt.subplots()
    ctaplot.plot_migration_matrix(recoX,
                                  recoY,
                                  colorbar=True,
                                  xy_line=False,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax
                                  )
    ax.scatter(np.mean(trueX), np.mean(trueY), color='red', marker='+', label='$DISP_{MC}$')
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disp_src.png'), facecolor='white', dpi=600)

    # Plot regresor and classifier
    fig, ax = plt.subplots()
    ctaplot.plot_roc_curve_gammaness(dl2.mc_type, dl2.gammaness, ax=ax)
    ax.set_title('ROC Curve : all energies')
    ax.set_ylabel(r'$\gamma$ True Positive Rate')
    ax.set_xlabel(r'$\gamma$ False Positive Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_all.png'), facecolor='white', dpi=600)

    fig, ax = plt.subplots(figsize=(8, 5))
    ctaplot.plot_roc_curve_gammaness_per_energy(dl2.mc_type,
                                                dl2.gammaness,
                                                dl2.mc_energy,
                                                ax=ax
                                                )
    box = (0.75, 0.4)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc="center left", ncol=1, bbox_to_anchor=box, frameon=True, fontsize=9)

    ax.set_title('ROC Curve : per energies bin')
    ax.set_ylabel(r'$\gamma$ True Positive Rate')
    ax.set_xlabel(r'$\gamma$ False Positive Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_per_energy_bin.png'), facecolor='white', dpi=600)

    h, l = ax.get_legend_handles_labels()

    irf = ctaplot.ana.irf_cta()
    energy_bins = irf.E_bin

    ebin_min = []
    ebin_max = []
    roc = []
    for line in l:
        bins = line.split('= ')[0].replace("]TeV - auc score", "").replace("[", "")

        ebin_min.append(float(bins.split(':')[0]))
        ebin_max.append(float(bins.split(':')[1]))
        roc.append(float(line.split('= ')[1]))

    ebin_min = np.array(ebin_min)
    ebin_max = np.array(ebin_max)
    roc = np.array(roc)

    df_roc = pd.DataFrame(
        {"ebin_min": ebin_min,
         "ebin_max": ebin_max,
         "mid_energy": 0.5 * (ebin_min + ebin_max),
         "ebin_min_true": np.zeros_like(ebin_min),
         "ebin_max_true": np.zeros_like(ebin_max),
         "de_min": np.zeros_like(ebin_min),
         "de_max": np.zeros_like(ebin_max),
         "roc": roc})

    for index, row in df_roc.iterrows():
        for j in range(len(energy_bins) - 1):
            if (row.mid_energy <= energy_bins[j + 1]) * (row.mid_energy > energy_bins[j]):
                print("{} < {} ≤ {}".format(energy_bins[j], row.mid_energy, energy_bins[j + 1]))
                row['ebin_min_true'] = energy_bins[j]
                row['ebin_max_true'] = energy_bins[j + 1]

                row['de_min'] = row['mid_energy'] - energy_bins[j]
                row['de_max'] = energy_bins[j + 1] - row['mid_energy']

    errors = [df_roc['de_min'], df_roc['de_max']]

    fig, ax = plt.subplots()
    ax.errorbar(df_roc['mid_energy'], df_roc['roc'], xerr=errors, fmt='o')
    ax.set_xscale('log')
    ax.grid()
    ax.set_xlim(energy_bins[0], energy_bins[-1])
    ax.set_ylim(0.2, 1.2)
    ax.set_ylabel('AUC ROC')
    ax.set_xlabel('$E_{RECO}$ [TeV]')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_per_energy_bin_2.png'), facecolor='white', dpi=600)

    config = read_configuration_file(config_file)
    reg_features_names = config['regression_features']
    clf_features_names = config['classification_features']

    energy = joblib.load(os.path.join(path_to_models, "reg_energy.sav"))
    disp = joblib.load(os.path.join(path_to_models, "reg_disp_vector.sav"))
    clf = joblib.load(os.path.join(path_to_models, "cls_gh.sav"))

    fig, ax = plt.subplots()
    plot_dl2.plot_importances(energy, reg_features_names, ax=ax)
    ax.set_title("Energy Regression")
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_reg.png'), facecolor='white', dpi=600)

    fig, ax = plt.subplots()
    plot_dl2.plot_importances(disp, reg_features_names, ax=ax)
    ax.set_title("DISP Regression")
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disp_reg.png'), facecolor='white', dpi=600)

    fig, ax = plt.subplots()
    plot_dl2.plot_importances(clf, clf_features_names, ax=ax)
    ax.set_title("$\gamma$/p classification")
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gh_class.png'), facecolor='white', dpi=600)

    fig, ax = plt.subplots()
    ax.hist(dl2[dl2['mc_type'] == 0]['gammaness'], bins=100, label='$\gamma$', alpha=0.65)
    ax.hist(dl2[dl2['mc_type'] == 101]['gammaness'], bins=100, label='p', alpha=0.65)
    ax.legend()
    ax.set_xlabel('gammaness')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gammaness.png'), facecolor='white', dpi=600)


if __name__ == '__main__':
    main()
