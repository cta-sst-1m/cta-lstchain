#!/bin/bash
#
# BASH script to launch jobs for converting dl1-test merged files to dl2 files, applying the RF method, reconstructing parameters
# ATTENTION :
# Gamma diffuse and electrons are out of the dl2 conversion. Electrons don't belong to the reconstruction model,
# Gamma diffuse are entirely for training, not for testing. Therefore, only gamma on source and protons are dl2-converted
#
# input_folder :        path to the dl1 file folder
# output_folder :       path to the folder where the dl2 files would be stored
# path_to_models :      path to the models
# json_config_file :    path to the json config file (lstchain)
# intensity :           cut in the minimum intensity value. Intensity goes from 0 to infinity
# leakage :             cut in the maximum leakage value. Leakage goes from 0 to 1.
# production_name :     just a name label for display output
# working_dir :         path to the folder where the mc_dl1_to_dl2.sh file is located
# cluster :             which cluster is used, e.g. itcluster, camk, yggdrasil
#

input_folder=$1
output_folder=$2
path_to_models=$3
json_config_file=$4
intensity=$5
leakage=$6
production_name=$7
working_dir=$8
cluster=$9

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  input_folder     : ${input_folder}"
echo "*-*-*-*-*-*-*-*-*-*  output_folder    : ${output_folder}"
echo "*-*-*-*-*-*-*-*-*-*  path_to_models   : ${path_to_models}"
echo "*-*-*-*-*-*-*-*-*-*  json_config_file : ${json_config_file}"
echo "*-*-*-*-*-*-*-*-*-*  intensity        : ${intensity}"
echo "*-*-*-*-*-*-*-*-*-*  leakage          : ${leakage}"
echo "*-*-*-*-*-*-*-*-*-*  production_name  : ${production_name}"
echo "*-*-*-*-*-*-*-*-*-*  working_dir      : ${working_dir}"
echo "*-*-*-*-*-*-*-*-*-*  cluster          : ${cluster}"
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

mkdir -p "${output_folder}"

for particle in "gamma_on" "gamma_off" "proton"; do

  echo " "
  echo ${particle}

  INPUT_FILE="${input_folder}/dl1_${particle}_merge_test.h5";
  OUTPUT_FOLDER=${output_folder}
  PATH_MODELS=${path_to_models}
  CONFIG_FILE=${json_config_file}
  INTENSITY=${intensity}
  LEAKAGE=${leakage}

  export INPUT_FILE=${INPUT_FILE}
  export OUTPUT_FOLDER=${OUTPUT_FOLDER}
  export PATH_MODELS=${PATH_MODELS}
  export CONFIG_FILE=${CONFIG_FILE}
  export INTENSITY=${INTENSITY}
  export LEAKAGE=${LEAKAGE}
  export working_dir=${working_dir}
  export particle=${particle}

  cd ${OUTPUT_FOLDER}

  if [[ "${cluster}" == "camk" ]]; then
    echo "Not implemented yet"
    sleep 0.2
  elif [[ "${cluster}" == "yggdrasil" ]]; then
    echo sbatch --job-name=dl2_${particle}_${INTENSITY}_${LEAKAGE} --output=dl2_${particle}.out --error=dl2_${particle}.err --export=INPUT_FILE=${INPUT_FILE},OUTPUT_FOLDER=${OUTPUT_FOLDER},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE},working_dir=${working_dir},particle=${particle} ${working_dir}/${cluster}.sbatch
    sbatch --job-name=dl2_${particle}_${INTENSITY}_${LEAKAGE} --output=dl2_${particle}.out --error=dl2_${particle}.err --export=INPUT_FILE=${INPUT_FILE},OUTPUT_FOLDER=${OUTPUT_FOLDER},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE},working_dir=${working_dir},particle=${particle} ${working_dir}/${cluster}.sbatch
    sleep 0.2
  elif [[ "${cluster}" == "itcluster" ]]; then
    echo sbatch --job-name=dl2_${particle}_${INTENSITY}_${LEAKAGE} --output=dl2_${particle}.out --error=dl2_${particle}.err --export=INPUT_FILE=${INPUT_FILE},OUTPUT_FOLDER=${OUTPUT_FOLDER},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE},working_dir=${working_dir},particle=${particle} ${working_dir}/${cluster}.sbatch
    sbatch --job-name=dl2_${particle}_${INTENSITY}_${LEAKAGE} --output=dl2_${particle}.out --error=dl2_${particle}.err --export=INPUT_FILE=${INPUT_FILE},OUTPUT_FOLDER=${OUTPUT_FOLDER},PATH_MODELS=${PATH_MODELS},CONFIG_FILE=${CONFIG_FILE},INTENSITY=${INTENSITY},LEAKAGE=${LEAKAGE},working_dir=${working_dir},particle=${particle} ${working_dir}/${cluster}.sbatch
    sleep 0.2
  else
    echo "unknown server, exiting"
    exit 1
  fi

done