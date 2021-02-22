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
path_to_lstchain="/home/david.miranda/software/cta-lstchain_v.${lstchain_version}"
cluster="itcluster"
rf_type="standard"
cam_key="dl1/event/telescope/parameters/LST_LSTCam"

percent_event_train=(30 50)
ARRAY_INTENSITY=(100 200)
ARRAY_LEAKAGE=(0.0 0.2)

for percentage in "${percent_event_train[@]}"; do

  train_percent=${percentage}
  test_percent=$((100 - train_percent))
  split_tag="${train_percent}.${test_percent}"

  working_dir="${path_to_lstchain}/lstchain/scripts/bash/rf_performance/${rf_type}/"

  for folder in "mono-lst-sipm-pmma-3ns" "mono-lst-sipm-pmma-5ns" "mono-lst-sipm-pmma-7ns"; do
#  for folder in "mono-lst-sipm-borofloat-3ns" "mono-lst-sipm-borofloat-5ns" "mono-lst-sipm-borofloat-7ns"; do
#  for folder in "tag_nominal_LST_09_2020_v2"; do
    for leakage in "${ARRAY_LEAKAGE[@]}"; do
      for intensity in "${ARRAY_INTENSITY[@]}"; do

        input_folder="/fefs/aswg/workspace/david.miranda/data/prod5/rf/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/rf_performances/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        production_name="Prod5_${folder}_with_lstchain_v.${lstchain_version}_leak_${leakage}_intensity_${intensity}"

        if [[ "${folder}" == "tag_nominal_LST_09_2020_v2" ]]; then
          json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/pmt_cam.json"
        else
          json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/sipm_cam.json"
        fi

        dl1_gamma_train="/fefs/aswg/workspace/david.miranda/data/prod5/dl1_merged/lstchain_v.${lstchain_version}/${folder}/${split_tag}/dl1_gamma_diffuse_merge_train.h5"
        dl1_gamma_test="/fefs/aswg/workspace/david.miranda/data/prod5/dl1_merged/lstchain_v.${lstchain_version}/${folder}/${split_tag}/dl1_gamma_on_merge_test.h5"
        dl1_proton_train="/fefs/aswg/workspace/david.miranda/data/prod5/dl1_merged/lstchain_v.${lstchain_version}/${folder}/${split_tag}/dl1_proton_merge_train.h5"
        dl1_proton_test="/fefs/aswg/workspace/david.miranda/data/prod5/dl1_merged/lstchain_v.${lstchain_version}/${folder}/${split_tag}/dl1_proton_merge_test.h5"

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
