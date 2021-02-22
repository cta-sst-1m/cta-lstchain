#!/bin/bash
#
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu,long,xxl

# script need following variables to be passed to sbatch:

# dl2_gamma_test
# dl2_proton_test
# output_dir
# max_gammaness
# gammanes_preselector
# intensity
# leakage
# magic_reference

# working_dir

source /fefs/aswg/software/virtual_env/anaconda3/bin/activate lst-dev
conda activate sipm_lstchain.0.6.3
echo "conda activate sipm_lstchain.0.6.3"

echo " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "

echo lstchain_unige_sensitivity --dl2_gamma_test $dl2_gamma_test --dl2_proton_test $dl2_proton_test --output_dir $output_dir --intensity_cut $intensity --leakage_cut $leakage --initial_gh_cut_efficiency $gammanes_preselector --max_gh_cut_efficiency $max_gammaness --magic_reference $magic_reference
lstchain_unige_sensitivity --dl2_gamma_test $dl2_gamma_test --dl2_proton_test $dl2_proton_test --output_dir $output_dir --intensity_cut $intensity --leakage_cut $leakage --initial_gh_cut_efficiency $gammanes_preselector --max_gh_cut_efficiency $max_gammaness --magic_reference $magic_reference