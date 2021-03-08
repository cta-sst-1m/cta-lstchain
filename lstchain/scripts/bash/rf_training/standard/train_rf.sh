#!/bin/bash
#
# BASH script to launch jobs training the RF model
# the RF models would be trained with diffuse gammas and protons in the standard RF training
# other RF models are not implemented yet
#
#
# input_folder :      path to the directory where the dl1 production is stored, i.e. the input directory
# output_folder :     path to the output directory
# json_config_file :  path to the json config file (lstchain)
# cam_key             camera key (from lstchain v.0.6.3, it is the same for either sipm or pmt cameras)
# intensity :         cut in the minimum intensity value. Intensity goes from 0 to infinity
# leakage :           cut in the maximum leakage value. Leakage goes from 0 to 1.
# production_name :   just a name label for display output
# working_dir :       path to the folder where the train_rf.sh file is located
# cluster :           name of the cluster that is used


input_folder=$1
output_folder=$2
json_config_file=$3
intensity=$4
leakage=$5
production_name=$6
working_dir=$7
cluster=$8

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  input_folder        : ${input_folder}"
echo "*-*-*-*-*-*-*-*-*-*  output_folder       : ${output_folder}"
echo "*-*-*-*-*-*-*-*-*-*  json_config_file    : ${json_config_file}"
echo "*-*-*-*-*-*-*-*-*-*  intensity           : ${intensity}"
echo "*-*-*-*-*-*-*-*-*-*  leakage             : ${leakage}"
echo "*-*-*-*-*-*-*-*-*-*  production_name     : ${production_name}"
echo "*-*-*-*-*-*-*-*-*-*  working_dir         : ${working_dir}"
echo "*-*-*-*-*-*-*-*-*-*  cluster             : ${cluster}"
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

mkdir -p "${output_folder}"

# training in gamma diffuse 100% and 50% of protons
# next step script would test in 100% gamma point source and and 50% of protons

GAMMAFILE="${input_folder}/dl1_gamma_diffuse_merge_train.h5"
PROTONFILE="${input_folder}/dl1_proton_merge_train.h5"
PATH_MODELS="${output_folder}"
CONFIG_FILE="${json_config_file}"
INTENSITY=${intensity}
LEAKAGE=${leakage}

export GAMMAFILE=${GAMMAFILE}
export PROTONFILE=${PROTONFILE}
export PATH_MODELS=${PATH_MODELS}
export CONFIG_FILE=${CONFIG_FILE}
export working_dir=${working_dir}
export INTENSITY=${INTENSITY}
export LEAKAGE=${LEAKAGE}

cd "${PATH_MODELS}"

if [[ "${cluster}" == "camk" ]]; then
  echo "Not yet implemented"
  sleep 0.2
elif [[ "${cluster}" == "yggdrasil" ]]; then
  echo "sbatch --job-name=train_job --output=train_job.out --error=train_job.err --export=GAMMAFILE=${GAMMAFILE},PROTONFILE=${PROTONFILE},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},working_dir=${working_dir},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE} ${working_dir}/${cluster}.sbatch"
  sbatch --job-name=train_job --output=train_job.out --error=train_job.err --export=GAMMAFILE=${GAMMAFILE},PROTONFILE=${PROTONFILE},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},working_dir=${working_dir},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE} ${working_dir}/${cluster}.sbatch
  sleep 0.2
elif [[ "${cluster}" == "itcluster" ]]; then
  echo "sbatch --job-name=train_job --output=train_job.out --error=train_job.err --export=GAMMAFILE=${GAMMAFILE},PROTONFILE=${PROTONFILE},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},working_dir=${working_dir},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE} ${working_dir}/${cluster}.sbatch"
  sbatch --job-name=train_job --output=train_job.out --error=train_job.err --export=GAMMAFILE=${GAMMAFILE},PROTONFILE=${PROTONFILE},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},working_dir=${working_dir},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE} ${working_dir}/${cluster}.sbatch
  sleep 0.2
else
  echo "unknown server, exiting"
  exit 1
fi
