#!/bin/bash
#
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu,long,xxl

# script need following variables to be passed to sbatch:

# json_config_file
# dl1_gamma_test
# dl1_gamma_train
# dl1_proton_test
# dl1_proton_train
# input_folder
# cam_key
# output_folder
# working_dir

source /fefs/aswg/software/virtual_env/anaconda3/bin/activate lst-dev
conda activate sipm_lstchain.0.6.3
echo "conda activate sipm_lstchain.0.6.3"

echo " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "

echo lstchain_unige_rf_performance --config_file $json_config_file --dl1_gamma_test $dl1_gamma_test --dl1_gamma_train $dl1_gamma_train --dl1_proton_test $dl1_proton_test --dl1_proton_train $dl1_proton_train --path_to_models $input_folder --dl1_cam_key $cam_key --output_dir $output_folder
lstchain_unige_rf_performance --config_file $json_config_file --dl1_gamma_test $dl1_gamma_test --dl1_gamma_train $dl1_gamma_train --dl1_proton_test $dl1_proton_test --dl1_proton_train $dl1_proton_train --path_to_models $input_folder --dl1_cam_key $cam_key --output_dir $output_folder