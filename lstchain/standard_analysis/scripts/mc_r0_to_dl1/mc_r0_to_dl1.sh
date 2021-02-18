#!/bin/bash
#
# BASH script to launch jobs for converting r0 (simtel) files to dl1 files, it uses default "lstchain_mc_r0_to_dl1" method
#
# input_folder :      path to the directory where the simtel production is saved for a given simtel configuration
# output_folder :     path to the directory where the dl1 files would be stored
# json_config_file :  path to the configuration file required by "lstchain_mc_r0_to_dl1"
# production_name :   name of the simtel configuration (for output display purpose)
# working_dir :       path to the folder where the mc_r0_to_dl1.sh file is located
# cluster :           name of the cluster that is used
#

input_folder=$1
output_folder=$2
json_config_file=$3
production_name=$4
working_dir=$5
cluster=$6

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  input_folder :       $input_folder "
echo "*-*-*-*-*-*-*-*-*-*  output_folder :      $output_folder "
echo "*-*-*-*-*-*-*-*-*-*  json_config_file :   $json_config_file "
echo "*-*-*-*-*-*-*-*-*-*  production_name :    $production_name "
echo "*-*-*-*-*-*-*-*-*-*  working_dir :        $working_dir "
echo "*-*-*-*-*-*-*-*-*-*  cluster :            $cluster "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

mkdir -p $dl1_data_dir
nfile_per_job=10

if [ ! -e $json_config_file ]; then
  echo "config file not found: $json_config_file"
  exit -1
fi

for particle in "proton" "gamma_on" "gamma_off" "gamma_diffuse" "electron"; do
  list_simtel_files="${input_folder}/${particle}_nsbx1/output/list.txt"

  if [[ -f $list_simtel_files ]]; then
    rm $list_simtel_files
  fi

  ls ${input_folder}/${particle}_nsbx1/output/*.simtel.gz | sort -V >$list_simtel_files
  n_simtel_files=$(cat $list_simtel_files | wc -l)
  echo "n_simtel_files: $n_simtel_files"
  n_jobs=$(($n_simtel_files / $nfile_per_job))
  if [ $(($n_jobs * $nfile_per_job)) -lt $n_simtel_files ]; then
    n_jobs=$(($n_jobs + 1))
  fi
  h5_dir="${output_folder}/${particle}"
  mkdir -p ${h5_dir}
  # If you want to submit a job array with a large number of samples, but you only want a few of them to run at a time,
  # you can specify this with a %. For example the following line will run an array with n_jobs jobs, but will only run 50 of them at a time
  # PBSARRAY="1-${n_jobs}%50"
  if [[ ${particle} == "gamma_on" ]]; then
    jobs_per_time=50
    PBSARRAY="1-${n_jobs}%${jobs_per_time}"
    SLURM_ARRAY="0-${n_jobs}%${jobs_per_time}"
    echo "${jobs_per_time} jobs per $particle at same time"
  elif [[ ${particle} == "gamma_off" ]]; then
    jobs_per_time=50
    PBSARRAY="1-${n_jobs}%${jobs_per_time}"
    SLURM_ARRAY="0-${n_jobs}%${jobs_per_time}"
    echo "${jobs_per_time} jobs per $particle at same time"
  elif [[ ${particle} == "gamma_diffuse" ]]; then
    jobs_per_time=50
    PBSARRAY="1-${n_jobs}%${jobs_per_time}"
    SLURM_ARRAY="0-${n_jobs}%${jobs_per_time}"
    echo "${jobs_per_time} jobs per $particle at same time"
  elif [[ ${particle} == "proton" ]]; then
    jobs_per_time=100
    PBSARRAY="1-${n_jobs}%${jobs_per_time}"
    SLURM_ARRAY="0-${n_jobs}%${jobs_per_time}"
    echo "${jobs_per_time} jobs per $particle at same time"
  elif [[ ${particle} == "electron" ]]; then
    jobs_per_time=50
    PBSARRAY="1-${n_jobs}%${jobs_per_time}"
    SLURM_ARRAY="0-${n_jobs}%${jobs_per_time}"
    echo "${jobs_per_time} jobs per $particle at same time"
  else
    jobs_per_time=50
    PBSARRAY="1-${n_jobs}%${jobs_per_time}"
    SLURM_ARRAY="0-${n_jobs}%${jobs_per_time}"
    echo "${jobs_per_time} jobs per $particle at same time"
  fi

  echo " "
  echo "/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/"
  echo "particle :        ${particle}"
  echo "input_folder :    ${input_folder}"
  echo "output_folder :   ${output_folder}"
  echo "h5_dir :          ${h5_dir}"
  echo "/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/"
  echo " "

  export h5_dir=${h5_dir}
  export list_simtel_files=${list_simtel_files}
  export json_config_file=${json_config_file}
  export nfile_per_job=${nfile_per_job}
  export working_dir=${working_dir}
  export cluster=${cluster}

  cd "${h5_dir}"
  if [[ "${cluster}" == "camk" ]]; then
    echo "qsub -t $PBSARRAY -o $h5_dir -e $h5_dir -v h5_dir=$h5_dir,list_simtel_files=$list_simtel_files,config_file=$json_config_file,nfile_per_job=$nfile_per_job -N dl0to1_${particle} ${working_dir}/${cluster}.job"
    qsub -t $PBSARRAY -o $h5_dir -e $h5_dir -v h5_dir=$h5_dir,list_simtel_files=$list_simtel_files,config_file=$json_config_file,nfile_per_job=$nfile_per_job -N dl0to1_${particle} ${working_dir}/${cluster}.job
    sleep 0.2
  elif [[ "${cluster}" == "yggdrasil" ]]; then
    echo "sbatch --array=${SLURM_ARRAY} --export=h5_dir=${h5_dir},list_simtel_files=${list_simtel_files},json_config_file=${json_config_file},nfile_per_job=${nfile_per_job} ${working_dir}/${cluster}.sbatch"
    sbatch --array=${SLURM_ARRAY} --export=h5_dir=${h5_dir},list_simtel_files=${list_simtel_files},json_config_file=${json_config_file},nfile_per_job=${nfile_per_job} ${working_dir}/${cluster}.sbatch
    sleep 0.2
  elif [[ "${cluster}" == "itcluster" ]]; then
    echo "sbatch --array=${SLURM_ARRAY} --export=h5_dir=${h5_dir},list_simtel_files=${list_simtel_files},json_config_file=${json_config_file},nfile_per_job=${nfile_per_job} ${working_dir}/${cluster}.sbatch"
    sbatch --array=${SLURM_ARRAY} --export=h5_dir=${h5_dir},list_simtel_files=${list_simtel_files},json_config_file=${json_config_file},nfile_per_job=${nfile_per_job} ${working_dir}/${cluster}.sbatch
    sleep 0.2
  else
    echo "unknown server, exiting"
    exit 1
  fi

done

