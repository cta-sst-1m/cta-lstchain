#!/bin/bash
#
# BASH script to launch jobs computing the performance of the RF model
#
# input_folder :      path to the RF trained models
# output_folder :     path to the output directory where plots would be stored
# json_config_file :  path to the json config file (lstchain)
# dl1_gamma_train :   path to the dl1 gamma file destined for training
# dl1_proton_train :  path to the dl1 proton file destined for training
# dl1_gamma_test :    path to the dl1 gamma file destined for training
# dl1_proton_test :   path to the dl1 proton file destined for testing
# cam_key :           camera key (from lstchain v.0.6.3, it is the same for either sipm or pmt cameras)
# intensity :         cut in the minimum intensity value. Intensity goes from 0 to infinity
# leakage :           cut in the maximum leakage value. Leakage goes from 0 to 1.
# production_name :   just a name label for display output
# working_dir :       path to the folder where the rf_performance.sh file is located
# cluster :           name of the cluster that is used

input_folder=$1
output_folder=$2
json_config_file=$3
dl1_gamma_train=$4
dl1_proton_train=$5
dl1_gamma_test=$6
dl1_proton_test=$7
cam_key=$8
intensity=$9
leakage=${10}
production_name=${11}
working_dir=${12}
cluster=${13}


echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  input_folder      :   ${input_folder}"
echo "*-*-*-*-*-*-*-*-*-*  output_folder     :   ${output_folder}"
echo "*-*-*-*-*-*-*-*-*-*  json_config_file  :   ${json_config_file}"
echo "*-*-*-*-*-*-*-*-*-*  dl1_gamma_train   :   ${dl1_gamma_train}"
echo "*-*-*-*-*-*-*-*-*-*  dl1_proton_train  :   ${dl1_proton_train}"
echo "*-*-*-*-*-*-*-*-*-*  dl1_gamma_test    :   ${dl1_gamma_test}"
echo "*-*-*-*-*-*-*-*-*-*  dl1_proton_test   :   ${dl1_proton_test}"
echo "*-*-*-*-*-*-*-*-*-*  cam_key           :   ${cam_key}"
echo "*-*-*-*-*-*-*-*-*-*  intensity         :   ${intensity}"
echo "*-*-*-*-*-*-*-*-*-*  leakage           :   ${leakage}"
echo "*-*-*-*-*-*-*-*-*-*  production_name   :   ${production_name}"
echo "*-*-*-*-*-*-*-*-*-*  working_dir       :   ${working_dir}"
echo "*-*-*-*-*-*-*-*-*-*  cluster           :   ${cluster}"
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

mkdir -p "${output_folder}"

export input_folder=${input_folder}
export output_folder=${output_folder}
export json_config_file=${json_config_file}
export dl1_gamma_train=${dl1_gamma_train}
export dl1_proton_train=${dl1_proton_train}
export dl1_gamma_test=${dl1_gamma_test}
export dl1_proton_test=${dl1_proton_test}
export cam_key=${cam_key}
export intensity=${intensity}
export leakage=${leakage}
export production_name=${production_name}
export working_dir=${working_dir}
export cluster=${cluster}

cd "${output_folder}"

if [[ "${cluster}" == "camk" ]]; then
  echo "Not implemented yet"
  sleep 0.2
elif [[ "${cluster}" == "yggdrasil" ]]; then
  echo "Not implemented yet"
  echo sbatch --job-name=rf_perform_${intensity}_${leakage} --output=rf_perform_${intensity}_${leakage}.out --error=rf_perform_${intensity}_${leakage}.err --export=json_config_file=${json_config_file},dl1_gamma_test=${dl1_gamma_test},dl1_gamma_train=${dl1_gamma_train},dl1_proton_test=${dl1_proton_test},dl1_proton_train=${dl1_proton_train},input_folder=${input_folder},cam_key=${cam_key},output_folder=${output_folder},working_dir=${working_dir} ${working_dir}/${cluster}.sbatch
  sbatch --job-name=rf_perform_${intensity}_${leakage} --output=rf_perform_${intensity}_${leakage}.out --error=rf_perform_${intensity}_${leakage}.err --export=json_config_file=${json_config_file},dl1_gamma_test=${dl1_gamma_test},dl1_gamma_train=${dl1_gamma_train},dl1_proton_test=${dl1_proton_test},dl1_proton_train=${dl1_proton_train},input_folder=${input_folder},cam_key=${cam_key},output_folder=${output_folder},working_dir=${working_dir} ${working_dir}/${cluster}.sbatch
  sleep 0.2
elif [[ "${cluster}" == "itcluster" ]]; then
  echo sbatch --job-name=rf_perform_${intensity}_${leakage} --output=rf_perform_${intensity}_${leakage}.out --error=rf_perform_${intensity}_${leakage}.err --export=json_config_file=${json_config_file},dl1_gamma_test=${dl1_gamma_test},dl1_gamma_train=${dl1_gamma_train},dl1_proton_test=${dl1_proton_test},dl1_proton_train=${dl1_proton_train},input_folder=${input_folder},cam_key=${cam_key},output_folder=${output_folder},working_dir=${working_dir} ${working_dir}/${cluster}.sbatch
  sbatch --job-name=rf_perform_${intensity}_${leakage} --output=rf_perform_${intensity}_${leakage}.out --error=rf_perform_${intensity}_${leakage}.err --export=json_config_file=${json_config_file},dl1_gamma_test=${dl1_gamma_test},dl1_gamma_train=${dl1_gamma_train},dl1_proton_test=${dl1_proton_test},dl1_proton_train=${dl1_proton_train},input_folder=${input_folder},cam_key=${cam_key},output_folder=${output_folder},working_dir=${working_dir} ${working_dir}/${cluster}.sbatch
  sleep 0.2
else
  echo "unknown server, exiting"
  exit 1
fi


