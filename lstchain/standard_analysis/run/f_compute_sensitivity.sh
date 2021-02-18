#!/bin/bash

# lstchain_version :      version of lstchain
# path_to_lstchain :      path to the installation of lstchain
# cluster :               which cluster is used, e.g. itcluster, camk, yggdrasil
# rf_type :               type of RF run for output directory naming/tracking (for now "standard")
#
# percent_event_train :   array with the different desired training percentages
# ARRAY_INTENSITY :       array of intensity to scan
# ARRAY_LEAKAGE :         array of leakage to scan
#
# input_folder :          path to the dl1 file folder
# output_folder :         path to the folder where the dl2 files would be stored
# path_to_models :        path to the models
# production_name :       just a name label for display output
# gammanes_preselector :  value of the first gammaness used to discriminate protons and gammas (unoptimized value)
# max_gammaness :         maximum value to scan gammaness (standard use 0.8, however, why limit us to this value and not 1)


lstchain_version="0.6.3"
path_to_lstchain="/home/david.miranda/software/cta-lstchain_v.${lstchain_version}"
cluster="itcluster"
rf_type="standard"

gammanes_preselector="0.5"
max_gammaness="1.0"

percent_event_train=(30 50)
ARRAY_INTENSITY=(100 200)
ARRAY_LEAKAGE=(0.0 0.2)

for percentage in "${percent_event_train[@]}"; do

  train_percent=${percentage}
  test_percent=$((100 - train_percent))
  split_tag="${train_percent}.${test_percent}"

  if [[ "${cluster}" == "camk" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/compute_sensitivity/${rf_type}/"
  elif [[ "${cluster}" == "yggdrasil" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/compute_sensitivity/${rf_type}/"
  elif [[ "${cluster}" == "itcluster" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/compute_sensitivity/${rf_type}/"
  else
    echo "unknown server, exiting"
    exit 1
  fi

  magic_reference="${path_to_lstchain}/lstchain/spectra/data/magic_sensitivity.txt"

  for folder in "mono-lst-sipm-pmma-3ns" "mono-lst-sipm-pmma-5ns" "mono-lst-sipm-pmma-7ns"; do
#  for folder in "mono-lst-sipm-borofloat-3ns" "mono-lst-sipm-borofloat-5ns" "mono-lst-sipm-borofloat-7ns"; do
#  for folder in "tag_nominal_LST_09_2020_v2"; do
    for leakage in "${ARRAY_LEAKAGE[@]}"; do
      for intensity in "${ARRAY_INTENSITY[@]}"; do

        dl2_gamma_test="/fefs/aswg/workspace/david.miranda/data/prod5/dl2/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/dl2_gamma_on_merge_test.h5"
        dl2_proton_test="/fefs/aswg/workspace/david.miranda/data/prod5/dl2/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/dl2_proton_merge_test.h5"
        output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/sensitivity/lstchain_v.${lstchain_version}/${folder}/${split_tag}/${rf_type}/intensity_${intensity}_leakage_${leakage}/"
        production_name="Prod5_${folder}_with_lstchain_v.${lstchain_version}_leak_${leakage}_intensity_${intensity}"

        bash ${working_dir}/sensitivity_computation.sh "${dl2_gamma_test}" "${dl2_proton_test}" "${output_folder}" "${intensity}" "${leakage}" "${gammanes_preselector}" "${max_gammaness}" "${magic_reference}" "${production_name}" "${working_dir}" "${cluster}"
        sleep 0.1

        echo ${dl2_gamma_test}
        echo ${dl2_proton_test}
        echo ${output_folder}
        echo ${intensity}
        echo ${leakage}
        echo ${gammanes_preselector}
        echo ${max_gammaness}
        echo ${magic_reference}
        echo ${production_name}
        echo ${working_dir}
        echo ${cluster}

      done
    done
  done
done
