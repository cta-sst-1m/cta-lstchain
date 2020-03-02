"""Pipeline for reconstruction of Energy, disp and gamma/hadron
separation of events stored in a simtelarray file.
Result is a dataframe with dl2 data.
Already trained Random Forests are required.

Usage:

$> python lst-recopipe arg1 arg2 ...

"""

import argparse
import os
import tables
import numpy as np
from astropy import units as u
from lstchain.io import read_configuration_file, standard_config, replace_config
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from ctapipe.image import tailcuts_clean


parser = argparse.ArgumentParser(description="Reconstruct events")

# Required arguments
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to a DL1 HDF5 file',
                    )

parser.add_argument('--outdir', '-o', action='store', type=str,
                     dest='outdir',
                     help='Path where to store the reco dl2 events',
                     default='./dl2_data')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--camera_name', '-cam', action='store', type=str,
                    dest='camera_name',
                    help='Name of the camera in the the dl1 file',
                    default='LSTCam'
                    )

args = parser.parse_args()


def camera_from_hdf5(datafile, camera_name):
    with tables.open_file(datafile, mode='r') as input:
        camera_info = input.root[
            f'instrument/telescope/camera/{camera_name}']
        pix_id = np.array([pix['pix_id'] for pix in camera_info])
        pix_x = np.array([pix['pix_x'] for pix in camera_info]) * u.m
        pix_y = np.array([pix['pix_y'] for pix in camera_info]) * u.m
        pix_area = np.array(
            [pix['pix_area'] for pix in camera_info]) * u.m * u.m
        return CameraGeometry(
            cam_id=camera_name, pix_id=pix_id, pix_x=pix_x, pix_y=pix_y,
            pix_area=pix_area, pix_type='hexagonal',
            pix_rotation="100d53m34.8s", cam_rotation="0d"
        )


def animate_datafile(datafile, config_file, camera_name):
    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(
                os.path.abspath(config_file))
        except:
            "Custom configuration could not be loaded !!!"
            exit()
    config = replace_config(standard_config, custom_config)
    cleaning_parameters = config["tailcut"]
    camera = camera_from_hdf5(datafile, camera_name)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
    display_charge = CameraDisplay(camera, ax=axes[0], autoupdate=True,
                                   autoscale=True, norm="log")
    display_charge.set_limits_minmax(1, 100)
    display_charge.add_colorbar(ax=axes[0])
    display_time = CameraDisplay(camera, ax=axes[1], autoupdate=True,
                                 autoscale=True)
    display_time.set_limits_minmax(0, 20)
    display_time.add_colorbar(ax=axes[1])
    plt.show(block=False)
    with tables.open_file(datafile, mode='r') as input:
        image_table = input.root[f'dl1/event/telescope/image/LST_{camera_name}']
        param_table = input.root[f'dl1/event/telescope/parameters/LST_{camera_name}']
        n_image = image_table.shape[0]
        assert n_image == param_table.shape[0]
        ellipse = None
        for row in range(n_image):
            charge = image_table[row]['image']
            signal_pixels = tailcuts_clean(camera, charge, **cleaning_parameters)
            display_charge.image = charge
            display_time.image = image_table[row]['pulse_time']
            centroid = (param_table[row]['x'], param_table[row]['y'])
            length = param_table[row]['length']
            width = param_table[row]['width']
            angle = param_table[row]['psi']
            intensity = param_table[row]['intensity']
            if ellipse is not None:
                ellipse.remove()
            display_charge.highlight_pixels(signal_pixels, color='r', linewidth=1, alpha=.5)
            display_time.highlight_pixels(signal_pixels, color='r', linewidth=1, alpha=.5)
            ellipse = display_charge.add_ellipse(centroid, 3*length, 3*width, angle, asymmetry=0.0)
            axes[0].set_title(f'intensity:{intensity} sum:{np.sum(charge[signal_pixels])}')
            plt.pause(1)


def plot_histo(datafile, camera_name):
    with tables.open_file(datafile, mode='r') as input:
        param_table = input.root[
            f'dl1/event/telescope/parameters/LST_{camera_name}'
        ]
        event_id = np.array(param_table.colinstances['event_id'])
        intensity = np.array(param_table.colinstances['intensity'])
        leakage = np.array(param_table.colinstances['leakage'])
        mc_energy = np.array(param_table.colinstances['mc_energy'])
        time_gradient = np.array(param_table.colinstances['time_gradient'])
        n_islands = np.array(param_table.colinstances['n_islands'])
        psi = np.array(param_table.colinstances['psi'])
        phi = np.array(param_table.colinstances['phi'])
        mc_core_distance = np.array(param_table.colinstances['mc_core_distance'])

        alpha = np.mod(psi - phi + np.pi/2, np.pi) - np.pi/2
        alpha_deg = alpha / np.pi * 180
        fig, axes = plt.subplots(2, 3, figsize=(12,8))
        axes[0][0].set_xscale('log')
        axes[0][0].hist2d(
            mc_energy, alpha_deg,
            bins=[
                np.logspace(np.log10(3e-3), np.log10(300), 30),
                np.linspace(-20, 20, 20)
            ],
            norm=mcolors.LogNorm()
        )
        axes[0][1].set_xscale('log')
        axes[0][1].set_yscale('log')
        axes[0][1].hist2d(
            mc_energy, intensity,
            bins=[
                np.logspace(np.log10(3e-3), np.log10(300), 30),
                np.logspace(np.log10(10), np.log10(1e5), 30),
            ],
            norm=mcolors.LogNorm()
        )
        axes[0][2].set_yscale('log')
        axes[0][2].hist2d(
            mc_core_distance, intensity,
            bins=[
                np.linspace(0, 1000, 20),
                np.logspace(np.log10(10), np.log10(1e5), 30),
            ],
            norm=mcolors.LogNorm()
        )
        filter = np.logical_and()
        axes[0][1].hist(intensity, 100, log=True)
        # axes[0][2].hist(leakage, 100, log=True)
        # axes[1][0].hist(mc_energy, 100, log=True)
        # axes[1][1].hist(time_gradient, 100, log=True)
        # axes[1][2].hist(n_islands, 100, log=True)
        plt.show()

def main():
    #plot_histo(args.datafile, args.camera_name)

    animate_datafile(args.datafile, args.config_file, args.camera_name)





if __name__ == '__main__':
    main()
