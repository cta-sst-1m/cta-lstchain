#!/usr/bin/env python

import argparse
import sys
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events
from lstchain.visualization import plot_dl2

import ctaplot
import joblib

from lstchain.io import (
    read_configuration_file,
    standard_config,
    replace_config,
)

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

parser.add_argument('--emin', type=float,
                    dest='emin', default=None,
                    help='lower bound of the energy bins')

parser.add_argument('--emax', type=float,
                    dest='emax', default=None,
                    help='higher bound of the energy bins')

parser.add_argument('--nbins', type=int,
                    dest='nbins', default=None,
                    help='Total number of energy bins in the emin to emax range')


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
    ax.set_yscale('log')
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
    nbins = args.nbins
    emin = args.emin
    emax = args.emax

    try:
        os.makedirs(output_dir, exist_ok=True)
        print("Directory ", output_dir, " Created")
    except FileExistsError:
        print("Directory ", output_dir, " already exists")

    custom_config = {}

    if config_file is not None:
        try:
            custom_config = read_configuration_file(config_file)
        except "Custom configuration could not be loaded!":
            pass

    config = replace_config(standard_config, custom_config)

    reg_energy = joblib.load(path_to_models + '/reg_energy.sav')
    reg_disp_vector = joblib.load(path_to_models + '/reg_disp_vector.sav')
    cls_gh = joblib.load(path_to_models + '/cls_gh.sav')

    dl2 = {}

    for a_set in ['train', 'test']:

        if a_set is 'train':
            dl1_gamma = dl1_gamma_train
            dl1_proton = dl1_proton_train
        elif a_set is 'test':
            dl1_gamma = dl1_gamma_test
            dl1_proton = dl1_proton_test
        else:
            print('wrong set!')
            sys.exit()

        gammas = filter_events(pd.read_hdf(dl1_gamma, key=dl1_cam_key), config["events_filters"])
        protons = filter_events(pd.read_hdf(dl1_proton, key=dl1_cam_key), config["events_filters"])

        data = pd.concat([gammas, protons], ignore_index=True)
        print('{} set'.format(a_set))
        print('Before drop NaN : ', len(data))
        data = data.dropna()
        print('After drop NaN : ', len(data))

        dl2_data = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_vector, custom_config=config)
        print('DL2 length : ', len(dl2_data))

        dl2[a_set] = dl2_data

        del gammas, protons, dl2_data

    selected_gammas_test = dl2['test'].query('reco_type==0 & mc_type==0')
    selected_gammas_train = dl2['train'].query('reco_type==0 & mc_type==0')

    if len(selected_gammas_test) == 0 or len(selected_gammas_train) == 0:
        print('No gammas selected, outputs will not be produced')
        sys.exit()

    if nbins is None:
        irf = ctaplot.ana.irf_cta()
        energy_bins = irf.E_bin
    else:
        energy_bins = np.logspace(np.log10(emin), np.log10(emax), nbins + 1)

    # Plot parameters
    for dataset in ['train', 'test']:
        for a_type in ['reco_type', 'mc_type']:
            if a_type is 'reco_type':
                true_particle = False
            elif a_type is 'mc_type':
                true_particle = True
            else:
                print('unknown type : mc_type or reco_type ?')
                exit(1)

            output_dir_type = os.path.join(output_dir, dataset, a_type)

            try:
                os.makedirs(output_dir_type, exist_ok=True)
                print("Directory ", output_dir_type, " Created")
            except FileExistsError:
                print("Directory ", output_dir_type, " already exists")

            plot_features(data=dl2[dataset],
                          intensity_cut=config['events_filters']['intensity'][0],
                          leakage_cut=config['events_filters']['leakage_intensity_width_2'][1],
                          r_cut=config['events_filters']['r'][1],
                          output_dir=output_dir_type,
                          true_particle=true_particle
                          )

    for a_set in ['train', 'test']:

        if a_set is 'train':
            selected_gammas = selected_gammas_train
        if a_set is 'test':
            selected_gammas = selected_gammas_test

        if os.path.exists(os.path.join(output_dir, 'e_reso_{}.h5'.format(a_set))):
            os.remove(os.path.join(output_dir, 'e_reso_{}.h5'.format(a_set)))
        if os.path.exists(os.path.join(output_dir, 'ang_reso_{}.h5'.format(a_set))):
            os.remove(os.path.join(output_dir, 'ang_reso_{}.h5'.format(a_set)))

        plot_dl2.energy_results(dl2_data=selected_gammas,
                                points_outfile=os.path.join(output_dir, 'e_reso_{}.h5'.format(a_set)),
                                plot_outfile=os.path.join(output_dir, 'e_reso_{}.png'.format(a_set))
                                )
        plt.close()
        plot_dl2.direction_results(dl2_data=selected_gammas,
                                   points_outfile=os.path.join(output_dir, 'ang_reso_{}.h5'.format(a_set)),
                                   plot_outfile=os.path.join(output_dir, 'ang_reso_{}.png'.format(a_set))
                                   )
        plt.close()

    # Plot Energy migration
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

    ctaplot.plot_migration_matrix(selected_gammas_train.mc_energy.apply(np.log10),
                                  selected_gammas_train.reco_energy.apply(np.log10),
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  ax=ax0)

    ctaplot.plot_migration_matrix(selected_gammas_test.mc_energy.apply(np.log10),
                                  selected_gammas_test.reco_energy.apply(np.log10),
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  ax=ax1)
    ax0.grid('both')
    ax1.grid('both')
    ax0.set_title('train')
    ax1.set_title('test')

    ax0.set_xlabel(r"$log_{10} E_{MC}$ [TeV]")
    ax1.set_xlabel(r"$log_{10} E_{MC}$ [TeV]")
    ax0.set_ylabel(r"$log_{10} E_{RECO}$ [TeV]")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e_migration.png'), facecolor='white')
    plt.close(fig)

    # Plot energy resolution and bias
    fig, ax = plt.subplots()
    ctaplot.plot_energy_resolution(selected_gammas_train.mc_energy,
                                   selected_gammas_train.reco_energy,
                                   ax=ax,
                                   bias_correction=False,
                                   color='tab:blue',
                                   fmt='o',
                                   label='train')

    ctaplot.plot_energy_resolution(selected_gammas_test.mc_energy,
                                   selected_gammas_test.reco_energy,
                                   ax=ax,
                                   bias_correction=False,
                                   color='tab:orange',
                                   fmt='^',
                                   label='test')

    ax.legend()
    ax.grid('on', which='both')
    ax.set_ylabel(r"${\Delta E/E}_{68}$")
    ax.set_xlabel(r"$E_{RECO} [TeV]$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e_reso.png'), facecolor='white', dpi=800)
    plt.close(fig)

    fig, ax = plt.subplots()
    ctaplot.plot_energy_bias(selected_gammas_train.mc_energy,
                             selected_gammas_train.reco_energy,
                             ax=ax,
                             fmt='o',
                             color='tab:blue',
                             label='train')

    ctaplot.plot_energy_bias(selected_gammas_test.mc_energy,
                             selected_gammas_test.reco_energy,
                             ax=ax,
                             fmt='^',
                             color='tab:orange',
                             label='test')

    ax.legend()
    ax.grid('on', which='both')
    ax.set_ylabel(r"Energy bias")
    ax.set_xlabel(r"$E_{RECO} [TeV]$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e_bias.png'), facecolor='white', dpi=800)
    plt.close(fig)

    # Plot theta2
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    ctaplot.plot_theta2(selected_gammas_train.reco_alt,
                        selected_gammas_train.reco_az,
                        selected_gammas_train.mc_alt,
                        selected_gammas_train.mc_az,
                        ax=ax0,
                        bins=100,
                        histtype='step',
                        color='tab:blue',
                        label='train',
                        range=(0, 1)
                        )

    ctaplot.plot_theta2(selected_gammas_test.reco_alt,
                        selected_gammas_test.reco_az,
                        selected_gammas_test.mc_alt,
                        selected_gammas_test.mc_az,
                        ax=ax1,
                        bins=100,
                        histtype='step',
                        color='tab:orange',
                        label='test',
                        range=(0, 1)
                        )

    ax0.legend()
    ax1.legend()
    ax0.grid('on', which='both')
    ax1.grid('on', which='both')
    ax0.set_xlabel(None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theta2.png'), facecolor='white', dpi=600)
    plt.close(fig)

    # Plot angular resolution
    fig, ax = plt.subplots()
    ctaplot.plot_angular_resolution_per_energy(selected_gammas_train.reco_alt,
                                               selected_gammas_train.reco_az,
                                               selected_gammas_train.mc_alt,
                                               selected_gammas_train.mc_az,
                                               selected_gammas_train.reco_energy,
                                               ax=ax,
                                               label='train',
                                               color='tab:blue',
                                               fmt='o'
                                               )

    ctaplot.plot_angular_resolution_per_energy(selected_gammas_test.reco_alt,
                                               selected_gammas_test.reco_az,
                                               selected_gammas_test.mc_alt,
                                               selected_gammas_test.mc_az,
                                               selected_gammas_test.reco_energy,
                                               ax=ax,
                                               label='test',
                                               color='tab:orange',
                                               fmt='^'
                                               )

    ax.legend()
    ax.set_xlabel(r"$E_{RECO} [TeV]$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angular_resolution.png'), facecolor='white', dpi=600)
    plt.close(fig)

    # Plot DISP vector
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True)

    ctaplot.plot_migration_matrix(selected_gammas_train.disp_dx,
                                  selected_gammas_train.reco_disp_dx,
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax0
                                  )

    ctaplot.plot_migration_matrix(selected_gammas_train.disp_dy,
                                  selected_gammas_train.reco_disp_dy,
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax1
                                  )

    ctaplot.plot_migration_matrix(selected_gammas_test.disp_dx,
                                  selected_gammas_test.reco_disp_dx,
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax2
                                  )

    ctaplot.plot_migration_matrix(selected_gammas_test.disp_dy,
                                  selected_gammas_test.reco_disp_dy,
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax3
                                  )

    for axs in [ax0, ax1]:
        axs.text(0.95, 0.09, 'train',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=axs.transAxes, bbox={'facecolor': 'tab:blue', 'alpha': 0.65})
    for axs in [ax2, ax3]:
        axs.text(0.95, 0.09, 'test',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=axs.transAxes, bbox={'facecolor': 'tab:orange', 'alpha': 0.65})
    ax2.set_xlabel('$DISP_{MC}$ [m]')
    ax3.set_xlabel('$DISP_{MC}$ [m]')
    ax0.set_ylabel('$DISP_{RECO}$ [m]')
    ax2.set_ylabel('$DISP_{RECO}$ [m]')
    ax0.set_title('DISP dx')
    ax1.set_title('DISP dy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disp_vector.png'), facecolor='white')
    plt.close(fig)

    # Plot DISP source
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

    trueX = selected_gammas_train[selected_gammas_train['mc_type'] == 0]['src_x']
    trueY = selected_gammas_train[selected_gammas_train['mc_type'] == 0]['src_y']
    recoX = selected_gammas_train[selected_gammas_train['reco_type'] == 0]['reco_src_x']
    recoY = selected_gammas_train[selected_gammas_train['reco_type'] == 0]['reco_src_y']

    ctaplot.plot_migration_matrix(recoX,
                                  recoY,
                                  colorbar=True,
                                  xy_line=False,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax0
                                  )

    ax0.scatter(np.mean(trueX), np.mean(trueY), color='red', marker='+', label='$DISP_{MC}$')
    del trueX, trueY, recoX, recoY

    trueX = selected_gammas_test[selected_gammas_test['mc_type'] == 0]['src_x']
    trueY = selected_gammas_test[selected_gammas_test['mc_type'] == 0]['src_y']
    recoX = selected_gammas_test[selected_gammas_test['reco_type'] == 0]['reco_src_x']
    recoY = selected_gammas_test[selected_gammas_test['reco_type'] == 0]['reco_src_y']

    ctaplot.plot_migration_matrix(recoX,
                                  recoY,
                                  colorbar=True,
                                  xy_line=False,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm(), bins=100),
                                  line_args=dict(color='black'),
                                  ax=ax1
                                  )

    ax1.scatter(np.mean(trueX), np.mean(trueY), color='red', marker='+', label='$DISP_{MC}$')
    del trueX, trueY, recoX, recoY

    ax0.legend()
    ax1.legend()
    ax0.set_title('train')
    ax1.set_title('test')
    ax0.set_xlabel('x [m]')
    ax1.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disp_src.png'), facecolor='white', dpi=600)
    plt.close(fig)

    # Plot AUC ROC
    fig, ax = plt.subplots()
    ctaplot.plot_roc_curve_gammaness(dl2['train'].mc_type, dl2['train'].gammaness, ax=ax)
    ctaplot.plot_roc_curve_gammaness(dl2['test'].mc_type, dl2['test'].gammaness, ax=ax)
    ax.set_title(None)
    ax.set_ylabel(r'$\gamma$ True Positive Rate')
    ax.set_xlabel(r'$\gamma$ False Positive Rate')

    h, l = ax.get_legend_handles_labels()
    l[0] = l[0].replace('auc score = ', 'train : ')
    l[1] = l[1].replace('auc score = ', 'test : ')
    ax.legend(handles=h, labels=l, title='Dataset : AUC Score', loc=4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc.png'), facecolor='white', dpi=600)
    plt.close(fig)

    # Plot ROC AUC curves
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    ctaplot.plot_roc_curve_gammaness_per_energy(dl2['train'].mc_type,
                                                dl2['train'].gammaness,
                                                dl2['train'].mc_energy,
                                                energy_bins=energy_bins,
                                                ax=ax0
                                                )

    ctaplot.plot_roc_curve_gammaness_per_energy(dl2['test'].mc_type,
                                                dl2['test'].gammaness,
                                                dl2['test'].mc_energy,
                                                energy_bins=energy_bins,
                                                ax=ax1
                                                )

    auc_score = []
    for axs in [ax0, ax1]:
        h, l = axs.get_legend_handles_labels()
        auc = []
        for i, label in enumerate(l):
            auc.append(l[i].split('= ')[1])
            l[i] = l[i].replace('[', '[ ').replace(']', ' ]').replace('TeV', '').replace(':', ' - ').replace(
                '- auc score = ', ' : ')
        auc_score.append(auc)
        if axs is ax0:
            title = 'Train - E bin [TeV] : AUC Score'
        elif axs is ax1:
            title = 'Test - E bin [TeV] : AUC Score'
        else:
            print('axis error')
            sys.exit()

        axs.legend(handles=h, labels=l, title=title, fontsize=10, loc=4)
        axs.set_title(None)
        axs.set_xlabel(r'$\gamma$ False Positive Rate')

    ax0.set_ylabel(r'$\gamma$ True Positive Rate')
    ax1.set_ylabel(None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_per_energy_bin.png'), facecolor='white', dpi=600)
    plt.close(fig)

    auc_score = np.array(auc_score, dtype=float)
    mid_energy = 0.5 * (energy_bins[:-1][1:] + energy_bins[:-1][:-1])
    de_min = mid_energy - energy_bins[:-1][:-1]
    de_max = energy_bins[:-1][1:] - mid_energy

    fig, ax = plt.subplots()
    ax.errorbar(mid_energy, auc_score[0], xerr=[de_min, de_max], fmt='o', color='tab:blue', label='train')
    ax.errorbar(mid_energy, auc_score[1], xerr=[de_min, de_max], fmt='^', color='tab:orange', label='test')
    ax.set_xscale('log')
    ax.grid()
    ax.legend()
    ax.set_xlim(emin, emax)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel('AUC ROC')
    ax.set_xlabel('$E_{RECO}$ [TeV]')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_vs_energy.png'), facecolor='white', dpi=600)
    plt.close(fig)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ax0.hist(dl2['train'][dl2['train']['mc_type'] == 0]['gammaness'], bins=100, label='$\gamma$', alpha=0.65)
    ax0.hist(dl2['train'][dl2['train']['mc_type'] == 101]['gammaness'], bins=100, label='p', alpha=0.65)
    ax0.legend(title='train')
    ax0.set_xlabel('gammaness')

    ax1.hist(dl2['test'][dl2['test']['mc_type'] == 0]['gammaness'], bins=100, label='$\gamma$', alpha=0.65)
    ax1.hist(dl2['test'][dl2['test']['mc_type'] == 101]['gammaness'], bins=100, label='p', alpha=0.65)
    ax1.legend(title='test')
    ax1.set_xlabel('gammaness')

    ax0.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gammaness.png'), facecolor='white', dpi=600)
    plt.close(fig)

    for share in [True, False]:
        for a_set in ['train', 'test']:
            fig, axs = plt.subplots(4, 5, sharex=True, sharey=share, figsize=(16, 7))
            axss = axs.ravel()
            df = dl2[a_set]
            for k in range(len(energy_bins) - 1):
                condition_energy = (df['reco_energy'] > energy_bins[k]) & (df['reco_energy'] <= energy_bins[k+1])
                condition_is_gamma = df['mc_type'] == 0
                condition_is_proton = df['mc_type'] == 101
                condition_on_gamma = condition_energy & condition_is_gamma
                condition_on_proton = condition_energy & condition_is_proton
                axss[k].hist(df[condition_on_gamma]['gammaness'], bins=100, alpha=0.65,
                             label=r'$\gamma$ : {}'.format(len(df[condition_on_gamma]['gammaness'])))
                axss[k].hist(df[condition_on_proton]['gammaness'], bins=100, alpha=0.65,
                             label='p : {}'.format(len(df[condition_on_proton]['gammaness'])))
                lower_bin = np.round(energy_bins[k], decimals=2)
                upper_bin = np.round(energy_bins[k + 1], decimals=2)
                axss[k].legend(title='{} - {} TeV'.format(lower_bin, upper_bin), fontsize=8, title_fontsize=10)
                axss[k].set_xlim(0.0 - 0.05, 1.0 + 0.05)
                axss[k].set_xlabel('gammaness')
            plt.tight_layout()
            if share:
                share_label = 'sharey_on'
            else:
                share_label = 'sharey_off'
            plt.savefig(os.path.join(output_dir, 'gammaness_ereco_bin_{}_{}.png'.format(a_set, share_label)),
                        facecolor='white', dpi=600)
            plt.close(fig)

    intensity_bins = np.linspace(np.min(df['log_intensity']), np.max(df['log_intensity']), 21)
    for share in [True, False]:
        for a_set in ['train', 'test']:
            fig, axs = plt.subplots(4, 5, sharex=True, sharey=share, figsize=(16, 7))
            axss = axs.ravel()
            df = dl2[a_set]
            for k in range(len(intensity_bins) - 1):
                condition_intensity = (df['log_intensity'] > intensity_bins[k]) & (
                        df['log_intensity'] <= intensity_bins[k + 1])
                condition_is_gamma = df['mc_type'] == 0
                condition_is_proton = df['mc_type'] == 101
                condition_on_gamma = condition_intensity & condition_is_gamma
                condition_on_proton = condition_intensity & condition_is_proton
                axss[k].hist(df[condition_on_gamma]['gammaness'], bins=100, alpha=0.65,
                             label=r'$\gamma$ : {}'.format(len(df[condition_on_gamma]['gammaness'])))
                axss[k].hist(df[condition_on_proton]['gammaness'], bins=100, alpha=0.65,
                             label='p : {}'.format(len(df[condition_on_proton]['gammaness'])))
                lower_bin = np.round(intensity_bins[k], decimals=2)
                upper_bin = np.round(intensity_bins[k + 1], decimals=2)
                axss[k].legend(title=r'$10^{{{}}}$ - $10^{{{}}}$ pe'.format(lower_bin, upper_bin), fontsize=8, title_fontsize=10)
                axss[k].set_xlabel('gammaness')
                axss[k].set_xlim(0.0 - 0.05, 1.0 + 0.05)
            plt.tight_layout()
            if share:
                share_label = 'sharey_on'
            else:
                share_label = 'sharey_off'
            plt.savefig(os.path.join(output_dir, 'gammaness_log_intensity_bin_{}_{}.png'.format(a_set, share_label)),
                        facecolor='white', dpi=600)
            plt.close(fig)

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
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_dl2.plot_importances(disp, reg_features_names, ax=ax)
    ax.set_title("DISP Regression")
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disp_reg.png'), facecolor='white', dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_dl2.plot_importances(clf, clf_features_names, ax=ax)
    ax.set_title("$\gamma$/p classification")
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gh_class.png'), facecolor='white', dpi=600)
    plt.close(fig)

    del dl2


if __name__ == '__main__':
    main()
