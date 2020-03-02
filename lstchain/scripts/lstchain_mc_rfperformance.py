"""Pipeline for test the performance of Random Forests.
4
Usage:

$>python lstchain_mc_rfperformance.py

"""

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events
from lstchain.visualization import plot_dl2
from lstchain.reco import utils
import astropy.units as u
from lstchain.io import standard_config, replace_config, read_configuration_file
from lstchain.io.io import dl1_params_lstcam_key

try:
    import ctaplot
except ImportError as e:
    print("ctaplot not installed, some plotting function will be missing")

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--gammafile', '-fg', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training')

parser.add_argument('--protonfile', '-fp', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training')

parser.add_argument('--gammatest', '-gt', type=str,
                    dest='gammatest',
                    help='path to the dl1 file of gamma events for test')

parser.add_argument('--protontest', '-pt', type=str,
                    dest='protontest',
                    help='path to the dl1 file of proton events for test')

# Optional arguments

parser.add_argument('--storerf', '-s', action='store', type=bool,
                    dest='storerf',
                    help='Boolean. True for storing trained RF in 3 files'
                    'Deafult=False, any user input will be considered True',
                    default=True)

parser.add_argument('--opath', '-o', action='store', type=str,
                     dest='path_models',
                     help='Path to store the resulting RF',
                     default='./saved_models/')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--cam_key', '-k', action='store', type=str,
                    dest='dl1_params_camera_key',
                    help='key to the camera table in the hdf5 files.',
                    default=dl1_params_lstcam_key
                    )

args = parser.parse_args()


def main():

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    reg_energy, reg_disp_vector, cls_gh = dl1_to_dl2.build_models(
        args.gammafile,
        args.protonfile,
        save_models=args.storerf,
        path_models=args.path_models,
        custom_config=config,
        dl1_params_camera_key=args.dl1_params_camera_key
    )

    gammas = filter_events(pd.read_hdf(args.gammatest, key=dl1_params_lstcam_key),
                           config["events_filters"],
                           )
    proton = filter_events(pd.read_hdf(args.protontest, key=dl1_params_lstcam_key),
                           config["events_filters"],
                           )

    data = pd.concat([gammas, proton], ignore_index=True)

    dl2 = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_vector, custom_config=config)

    ####PLOT SOME RESULTS#####

    gammas = dl2[dl2.gammaness >= 0.5]
    protons = dl2[dl2.gammaness < 0.5]
    gammas.reco_type = 0
    protons.reco_type = 1

    focal_length = 28 * u.m
    src_pos_reco = utils.reco_source_position_sky(gammas.x.values * u.m,
                                                  gammas.y.values * u.m,
                                                  gammas.reco_disp_dx.values * u.m,
                                                  gammas.reco_disp_dy.values * u.m,
                                                  focal_length,
                                                  gammas.mc_alt_tel.values * u.rad,
                                                  gammas.mc_az_tel.values * u.rad)


    plot_dl2.plot_features(dl2)
    plt.savefig(args.path_models + '/histograms.png')

    plot_dl2.plot_e(gammas, 10, 1.5, 3.5)
    plt.savefig(args.path_models + '/energy_reco_gamma.png')

    plot_dl2.calc_resolution(gammas)
    plt.savefig(args.path_models + '/resolution_gamma.png')

    plot_dl2.plot_e_resolution(gammas, 10, 1.5, 3.5)
    plt.savefig(args.path_models + '/energy_resolution_gamma.png')

    plot_dl2.plot_disp_vector(gammas)
    plt.savefig(args.path_models + '/disp_reco_gamma.png')


    try:
        ctaplot.plot_theta2(gammas.mc_alt,
                            np.arctan(np.tan(gammas.mc_az)),
                            src_pos_reco.alt.rad,
                            np.arctan(np.tan(src_pos_reco.az.rad)),
                            bins=50, range=(0, 1),
        )
        plt.savefig(args.path_models + '/theta2_gamma.png')
        ctaplot.plot_angular_res_per_energy(src_pos_reco.alt.rad,
                                            np.arctan(np.tan(src_pos_reco.az.rad)),
                                            gammas.mc_alt,
                                            np.arctan(np.tan(gammas.mc_az)),
                                            gammas.mc_energy
        )
        plt.savefig(args.path_models + '/angular_resolution_gamma.png')
    except:
        pass

    regression_features = config["regression_features"]
    classification_features = config["classification_features"]

    plot_dl2.plot_pos(dl2)
    plt.savefig(args.path_models + '/position_reco.png')
    plot_dl2.plot_ROC(cls_gh, dl2, classification_features, -1)
    plt.savefig(args.path_models + '/roc.png')
    plot_dl2.plot_importances(cls_gh, classification_features)
    plt.savefig(args.path_models + '/features_gh.png')
    plot_dl2.plot_importances(reg_energy, regression_features)
    plt.savefig(args.path_models + '/features_energy_reco.png')
    plot_dl2.plot_importances(reg_disp_vector, regression_features)
    plt.savefig(args.path_models + '/features_disp_reco.png')

    plt.hist(dl2[dl2['mc_type']==101]['gammaness'], bins=100, label="proton")
    plt.hist(dl2[dl2['mc_type']==0]['gammaness'], bins=100, label="gamma")
    plt.xlabel('gammaness')
    plt.ylabel('counts')
    plt.legend()
    plt.savefig(args.path_models + '/gammaness.png')



if __name__ == '__main__':
    main()
