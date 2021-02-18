#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -q qlong
#PBS -l vmem=4000mb

# needed variables:
# run_dir
# merged_file

source /work/se1/unige/software/SetupPackages.sh
conda activate lst-dev-our_MC

lstchain_merge_hdf5_files -d $run_dir -o $merged_file --smart False --no-image True &> ${merged_file/.h5/.out}
