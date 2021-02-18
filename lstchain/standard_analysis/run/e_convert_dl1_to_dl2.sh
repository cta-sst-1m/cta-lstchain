#!/bin/bash
#
# lstchain_version :    version of lstchain
# path_to_lstchain :    path to the installation of lstchain
# cluster :             which cluster is used, e.g. itcluster, camk, yggdrasil
# rf_type :             type of RF run for output directory naming/tracking (for now "standard")
#
# percent_event_train : array with the different desired training percentages
# ARRAY_INTENSITY :     array of intensity to scan
# ARRAY_LEAKAGE :       array of leakage to scan
#
# input_folder :        path to the dl1 file folder
# output_folder :       path to the folder where the dl2 files would be stored
# path_to_models :      path to the models
# json_config_file :    path to the json config file (lstchain)
# production_name :     just a name label for display output

lstchain_version="0.6.3"
path_to_lstchain="/home/david.miranda/software/cta-lstchain_v.${lstchain_version}"
cluster="itcluster"
rf_type="standard"

percent_event_train=(30 50)
ARRAY_INTENSITY=(100 200)
ARRAY_LEAKAGE=(0.0 0.2)

for percentage in "${percent_event_train[@]}"; do

  train_percent=${percentage}
  test_percent=$((100 - train_percent))
  split_tag="${train_percent}.${test_percent}"

  if [[ "${cluster}" == "camk" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/mc_dl1_to_dl2/"
  elif [[ "${cluster}" == "yggdrasil" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/mc_dl1_to_dl2/"
  elif [[ "${cluster}" == "itcluster" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/mc_dl1_to_dl2/"
  else
    echo "unknown server, exiting"
    exit 1
  fi

  for folder in "mono-lst-sipm-pmma-3ns" "mono-lst-sipm-pmma-5ns" "mono-lst-sipm-pmma-7ns"; do
#  for folder in "mono-lst-sipm-borofloat-3ns" "mono-lst-sipm-borofloat-5ns" "mono-lst-sipm-borofloat-7ns"; do
#  for folder in "tag_nominal_LST_09_2020_v2"; do
    for leakage in "${ARRAY_LEAKAGE[@]}"; do
      for intensity in "${ARRAY_INTENSITY[@]}"; do

        if [[ "${folder}" == "tag_nominal_LST_09_2020_v2" ]]; then
          json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/pmt_cam.json"
        else
          json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/sipm_cam.json"
        fi

        path_to_models="/fefs/aswg/workspace/david.miranda/data/prod5/rf/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        input_folder="/fefs/aswg/workspace/david.miranda/data/prod5/dl1_merge/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/dl2/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        production_name="Prod5_${folder}_with_lstchain_v.${lstchain_version}_leak_${leakage}_intensity_${intensity}"

        bash ${working_dir}/mc_dl1_to_dl2.sh "${input_folder}" "${output_folder}" "${path_to_models}" "${json_config_file}" "${intensity}" "${leakage}" "${production_name}" "${working_dir}" "${cluster}"
        sleep 0.1

        echo ${input_folder}
        echo ${output_folder}
        echo ${path_to_models}
        echo ${json_config_file}
        echo ${intensity}
        echo ${leakage}
        echo ${production_name}
        echo ${working_dir}
        echo ${cluster}

    done
  done
done
