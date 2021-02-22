#!/bin/bash
#
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu,long,xxl

# script need following variables to be passed to sbatch:
# GAMMAFILE
# PROTONFILE
# PATH_MODELS
# CONFIG_FILE
# DL1_PARAMS_CAMERA_KEY
# working_dir
# INTENSITY
# LEAKAGE


source /fefs/aswg/software/virtual_env/anaconda3/bin/activate lst-dev
conda activate sipm_lstchain.0.6.3
echo "conda activate sipm_lstchain.0.6.3"

echo " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "

echo "lstchain_unige_training --input-file-gamma $GAMMAFILE --input-file-proton $PROTONFILE --output-dir $PATH_MODELS --config $CONFIG_FILE --cam_key $DL1_PARAMS_CAMERA_KEY --intensity $INTENSITY --leakage_intensity_width_2 $LEAKAGE"
lstchain_unige_training --input-file-gamma $GAMMAFILE --input-file-proton $PROTONFILE --output-dir $PATH_MODELS --config $CONFIG_FILE --cam_key $DL1_PARAMS_CAMERA_KEY --intensity $INTENSITY --leakage_intensity_width_2 $LEAKAGE