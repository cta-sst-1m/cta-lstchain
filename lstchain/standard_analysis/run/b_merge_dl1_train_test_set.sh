#!/bin/bash

# lstchain_version :    version of lstchain
# path_to_lstchain :    path to the installation of lstchain
# cluster :             which cluster is used, e.g. itcluster, camk, yggdrasil
# percent_event_train : array with the different desired training percentages

# input_folder :        folder with the dl1 files
# output_folder :       folder with the merged dl1 files
# production_name :     just a name label for display output

lstchain_version="0.6.3"
path_to_lstchain="/home/david.miranda/software/cta-lstchain_v.${lstchain_version}"
cluster="itcluster"
percent_event_train=(30 50)

for percentage in "${percent_event_train[@]}"; do
#  for folder in "tag_nominal_LST_09_2020_v2"; do
#  for folder in "mono-lst-sipm-borofloat-3ns" "mono-lst-sipm-borofloat-5ns" "mono-lst-sipm-borofloat-7ns"; do
  for folder in "mono-lst-sipm-pmma-3ns" "mono-lst-sipm-pmma-5ns" "mono-lst-sipm-pmma-7ns"; do

    train_percent=${percentage}
    test_percent=$((100-train_percent))
    split_tag="${train_percent}.${test_percent}"

    input_folder="/fefs/aswg/workspace/david.miranda/data/prod5/dl1/lstchain_v.${lstchain_version}/${folder}/"
    output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/dl1_merged/lstchain_v.${lstchain_version}/${folder}/${split_tag}"
    production_name="Prod5 ${folder} with lstchain_v.${lstchain_version}"

    cluster="itcluster"

    if [[ "${cluster}" == "camk" ]]; then
      echo "Not implemented yet"
      working_dir=" "
    elif [[ "${cluster}" == "yggdrasil" ]]; then
      echo "Not implemented yet"
      working_dir=" "
    elif [[ "${cluster}" == "itcluster" ]]; then
      working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/merge_dl1_train_test_set/"
    else
      echo "unknown server, exiting"
      exit 1
    fi

    bash ${working_dir}/merge_dl1_train_test_set.sh "${input_folder}" "${output_folder}" "${percentage}" "${production_name}" "${working_dir}" "${cluster}"

    sleep 0.1

    echo ${input_folder}
    echo ${output_folder}
    echo ${percentage}
    echo ${production_name}
    echo ${working_dir}
    echo ${cluster}

  done
done
