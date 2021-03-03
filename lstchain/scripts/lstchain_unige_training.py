#!/usr/bin/env python

import os
import json
import argparse
from lstchain.reco import dl1_to_dl2
from distutils.util import strtobool
from lstchain.io.config import read_configuration_file
from lstchain.io.io import dl1_params_lstcam_key

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--input-file-gamma', '--fg', type=str,
                    dest='gammafile',
                    help='Path to the dl1 file of gamma events for training')

parser.add_argument('--input-file-proton', '--fp', type=str,
                    dest='protonfile',
                    help='Path to the dl1 file of proton events for training')

# Optional arguments
parser.add_argument('--store-rf', '-s', action='store', type=lambda x: bool(strtobool(x)),
                    dest='storerf',
                    help='Boolean. True for storing trained models in 3 files'
                         'Default=True, use False otherwise',
                    default=True)

parser.add_argument('--output-dir', '-o', action='store', type=str,
                    dest='path_models',
                    help='Path to store the resulting RF',
                    default='./trained_models/')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--cam_key', '-k', action='store', type=str,
                    dest='dl1_params_camera_key',
                    help='key to the camera table in the hdf5 files.',
                    default=dl1_params_lstcam_key
                    )

parser.add_argument('--intensity', '-i', action='store', type=float,
                    dest='intensity',
                    help='minimum intensity that could be applied',
                    default=None
                    )

parser.add_argument('--width', '-w', action='store', type=float,
                    dest='width',
                    help='maximum width',
                    default=None
                    )

parser.add_argument('--length', '-l', action='store', type=float,
                    dest='length',
                    help='maximum length',
                    default=None
                    )

parser.add_argument('--wl', action='store', type=float,
                    dest='wl',
                    help='minimum intensity that could be applied',
                    default=None
                    )

parser.add_argument('--radius', action='store', type=float,
                    dest='radius',
                    help='maximum radius that could be applied',
                    default=None
                    )

parser.add_argument('--leakage_intensity_width_1', action='store', type=float,
                    dest='leakage_intensity_width_1',
                    help='leakage value, a float between 0 and 1. It represents a fraction',
                    default=None
                    )

parser.add_argument('--leakage_intensity_width_2', action='store', type=float,
                    dest='leakage_intensity_width_2',
                    help='leakage value, a float between 0 and 1. It represents a fraction',
                    default=None
                    )

parser.add_argument('--leakage_pixel_width_1', action='store', type=float,
                    dest='leakage_pixel_width_1',
                    help='leakage value, a float between 0 and 1. It represents a fraction',
                    default=None
                    )

parser.add_argument('--leakage_pixel_width_2', action='store', type=float,
                    dest='leakage_pixel_width_2',
                    help='leakage value, a float between 0 and 1. It represents a fraction',
                    default=None
                    )

args = parser.parse_args()


def main():
    # Train the models

    config = {}

    if args.config_file is not None:
        try:
            config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    if args.intensity is not None:
        config['events_filters']['intensity'][0] = args.intensity

    if args.width is not None:
        config['events_filters']['width'][1] = args.width

    if args.length is not None:
        config['events_filters']['length'][1] = args.length

    if args.wl is not None:
        config['events_filters']['wl'][1] = args.wl
    else:
        config['events_filters']['wl'][1] = 1.0

    if args.radius is not None:
        config['events_filters']['r'][1] = args.radius
    else:
        config['events_filters']['r'][1] = 1.0

    if args.leakage_intensity_width_1 is not None:
        config['events_filters']['leakage_intensity_width_1'][1] = args.leakage_intensity_width_1

    if args.leakage_intensity_width_2 is not None:
        config['events_filters']['leakage_intensity_width_2'][1] = args.leakage_intensity_width_2

    if args.leakage_pixel_width_1 is not None:
        config['events_filters']['leakage_pixel_width_1'][1] = args.leakage_pixel_width_1

    if args.leakage_pixel_width_2 is not None:
        config['events_filters']['leakage_pixel_width_2'][1] = args.leakage_pixel_width_2

    with open(os.path.join(args.path_models, 'config.json'), "w") as f:
        json.dump(config, f, indent=2)

    print("Configuration file used :")
    print(config)

    dl1_to_dl2.build_models(args.gammafile,
                            args.protonfile,
                            save_models=args.storerf,
                            path_models=args.path_models,
                            custom_config=config,
                            dl1_params_camera_key=args.dl1_params_camera_key
                            )



if __name__ == '__main__':
    main()