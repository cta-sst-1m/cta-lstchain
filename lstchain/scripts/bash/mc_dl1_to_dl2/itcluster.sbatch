#!/bin/bash
#
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=long,xxl,gpu

# script need following variables to be passed to sbatch:
# INPUT_FILE
# OUTPUT_FOLDER
# PATH_MODELS
# CONFIG_FILE
# working_dir
# INTENSITY
# LEAKAGE
# particle

source /fefs/aswg/software/virtual_env/anaconda3/bin/activate lst-dev

echo " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "

conda activate sipm_lstchain.0.6.3
echo "conda activate sipm_lstchain.0.6.3"

new_config="$OUTPUT_FOLDER/${particle}.json"
echo "new config file : $new_config"

now=$(date +"%T")
echo "Current time : $now"

# Creates a json config file from original file with modified event filters
echo lstchain_unige_dump_config --config_file $CONFIG_FILE --output_file $new_config --intensity $INTENSITY --leakage_intensity_width_2 $LEAKAGE
lstchain_unige_dump_config --config_file $CONFIG_FILE --output_file $new_config --intensity $INTENSITY --leakage_intensity_width_2 $LEAKAGE
sleep 0.5

echo lstchain_dl1_to_dl2 --input-file $INPUT_FILE --path-models $PATH_MODELS --output-dir $OUTPUT_FOLDER --config $new_config
lstchain_dl1_to_dl2 --input-file $INPUT_FILE --path-models $PATH_MODELS --output-dir $OUTPUT_FOLDER --config $new_config

then=$(date +"%T")
echo "Current time : $then"

