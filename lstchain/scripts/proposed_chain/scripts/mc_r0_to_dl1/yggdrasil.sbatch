#!/bin/bash -l
#
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --partition=dpnc-private-gpu
#SBATCH --job-name=dl0_to_dl1-%a
#SBATCH --output=dl0_to_dl1-%A-%a.out
#SBATCH --error=dl0_to_dl1-%A-%a.err

# script need folowing variables to be passed to sbatch:
# SLURM_ARRAY_TASK_ID (set by PBS when called with -t )
# nfile_per_job
# list_simtel_files
# h5_dir
# json_config_file

conda activate sipm_lstchain.0.6.3
echo "conda activate sipm_lstchain.0.6.3"

echo "${nfile_per_job}"
echo "${list_simtel_files}"
echo "${h5_dir}"
echo "${json_config_file}"

echo "list_simtel_files: $list_simtel_files"
simtel_files=( $(cat $list_simtel_files) )
index_job=$(( $SLURM_ARRAY_TASK_ID - 1 ))
first_job=$(( $index_job * $nfile_per_job ))
last_job=$(( $(( $index_job+1 )) * $nfile_per_job - 1 ))
if [ $last_job -gt ${#simtel_files[@]} ]; then
    last_job=${#simtel_files[@]}
fi
for i in $(seq $first_job $last_job); do
    simtel_file=${simtel_files[$i]}
    echo "simtel_file: $simtel_file"
    file_basename="$(basename $simtel_file)"
    output_file="$h5_dir/dl1_${file_basename/.gz/.h5}"
    if [ -e $output_file ]; then
        echo "$output_file exists, skipping conversion to DL1"
        exit 0
    fi
    echo "lstchain_mc_r0_to_dl1 --input-file $simtel_file --output-dir $h5_dir --config $json_config_file"
    lstchain_mc_r0_to_dl1 --input-file $simtel_file --output-dir $h5_dir --config $json_config_file

done
