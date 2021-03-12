#!/bin/bash

# lstchain_version :    version of lstchain
# path_to_lstchain :    path to the installation of lstchain
# cluster :             which cluster is used, e.g. itcluster, camk, yggdrasil
# rf_type :             type of RF run for output directory naming/tracking (for now "standard")
# cam_key :             camera key (from lstchain v.0.6.3, it is the same for either sipm or pmt cameras)

# percent_event_train : array with the different desired training percentages
# ARRAY_INTENSITY :     array of intensity to scan
# ARRAY_LEAKAGE :       array of leakage to scan

# input_folder :        path to the models
# output_folder :       path to the output plots
# json_config_file :    path to the json config file (lstchain)
# production_name :     just a name label for display output

lstchain_version="0.6.3"
path_to_lstchain="/home/david.miranda/software/cta-sipm-lstchain_v.${lstchain_version}"
cluster="itcluster"
rf_type="standard"
cam_key="dl1/event/telescope/parameters/LST_LSTCam"

# This values was configurable to see if the split percentage has a impact in the performances of the RF training
ptag=80

percent_event_train=(30 50)
ARRAY_INTENSITY=(0)
ARRAY_LEAKAGE=(1.0)

for percentage in "${percent_event_train[@]}"; do

  train_percent=${percentage}
  test_percent=$((100 - train_percent))
  split_tag="${train_percent}.${test_percent}"

  working_dir="${path_to_lstchain}/lstchain/scripts/bash/rf_performance/${rf_type}/"

  for folder in "tag_nominal_LST_09_2020_v2" "mono-lst-sipm-pmma-3ns" "mono-lst-sipm-pmma-5ns" "mono-lst-sipm-pmma-7ns" "mono-lst-sipm-borofloat-3ns" "mono-lst-sipm-borofloat-5ns" "mono-lst-sipm-borofloat-7ns"; do
    for leakage in "${ARRAY_LEAKAGE[@]}"; do
      for intensity in "${ARRAY_INTENSITY[@]}"; do

        input_folder="/fefs/aswg/workspace/david.miranda/data/prod5/rf/lstchain_v.${lstchain_version}_${ptag}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/rf_performances/lstchain_v.${lstchain_version}_${ptag}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        production_name="Prod5_${folder}_with_lstchain_v.${lstchain_version}_leak_${leakage}_intensity_${intensity}_${ptag}"

        if [[ "${folder}" == "tag_nominal_LST_09_2020_v2" ]]; then
          json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/pmt_cam.json"
        else
          json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/sipm_cam.json"
        fi

        dl1_gamma_train="${input_folder}/gamma_train.h5"
        dl1_gamma_test="${input_folder}/gamma_test.h5"
        dl1_proton_train="${input_folder}/proton_train.h5"
        dl1_proton_test="${input_folder}/proton_test.h5"

        bash ${working_dir}/rf_performance.sh "${input_folder}" "${output_folder}" "${json_config_file}" "${dl1_gamma_train}" "${dl1_proton_train}" "${dl1_gamma_test}" "${dl1_proton_test}" "${cam_key}" "${intensity}" "${leakage}" "${production_name}" "${working_dir}" "${cluster}"
        sleep 0.1

        echo ${input_folder}
        echo ${output_folder}
        echo ${json_config_file}
        echo ${dl1_gamma_train}
        echo ${dl1_proton_train}
        echo ${dl1_gamma_test}
        echo ${dl1_proton_test}
        echo ${cam_key}
        echo ${intensity}
        echo ${leakage}
        echo ${production_name}
        echo ${working_dir}
        echo ${cluster}

      done
    done
  done
done
