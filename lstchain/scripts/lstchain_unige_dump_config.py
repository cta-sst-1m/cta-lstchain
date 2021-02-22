#!/usr/bin/env python

import os
import json
import argparse
from lstchain.io import (
    read_configuration_file,
    standard_config,
    replace_config,
    write_dl2_dataframe,
    get_dataset_keys,
)

parser = argparse.ArgumentParser(description="Dumping a config file")

# Required arguments
parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    required=True
                    )

parser.add_argument('--output_file', '-o', action='store', type=str,
                     dest='output_file',
                     help='Path where to store the modified config file',
                     required=True
                    )

# Optional arguments level
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

    config_file = args.config_file
    output_file = args.output_file

    config = {}

    if args.config_file is not None:
        try:
            config = read_configuration_file(config_file)
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


    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
