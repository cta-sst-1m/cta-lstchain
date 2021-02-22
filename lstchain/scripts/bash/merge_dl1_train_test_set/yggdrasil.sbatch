#!/bin/bash
#
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --partition=private-dpnc-gpu

# script need folowing variables to be passed to sbatch:
# SLURM_ARRAY_TASK_ID (set by PBS when called with -t )
# run_dir
# merged_file

source /fefs/aswg/software/virtual_env/anaconda3/bin/activate lst-dev

conda activate sipm_lstchain.0.6.3
echo "conda activate sipm_lstchain.0.6.3"

echo " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "

echo "lstchain_merge_hdf5_files --input-dir $run_dir --output-file $merged_file --smart False --no-image True &> ${merged_file/.h5/.out}"
lstchain_merge_hdf5_files --input-dir $run_dir --output-file $merged_file --smart False --no-image True &> ${merged_file/.h5/.out}


