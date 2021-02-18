#!/bin/bash

# lstchain_version : version of lstchain
# path_to_lstchain : path to the installation of lstchain
# cluster :          which cluster is used, e.g. itcluster, camk, yggdrasil

# input_folder :     folder with the r0 (simtel) files
# output_folder :    folder with the dl1 files
# production_name :  just a name label for display output
# json_config_file : path to the json config file (lstchain)

lstchain_version="0.6.3"
path_to_lstchain="/home/david.miranda/software/cta-lstchain_v.${lstchain_version}"
cluster="itcluster"

#for folder in "tag_nominal_LST_09_2020_v2"; do
#for folder in "mono-lst-sipm-borofloat-3ns" "mono-lst-sipm-borofloat-5ns" "mono-lst-sipm-borofloat-7ns"; do
for folder in "mono-lst-sipm-pmma-3ns" "mono-lst-sipm-pmma-5ns" "mono-lst-sipm-pmma-7ns"; do

  input_folder="/fefs/aswg/workspace/david.miranda/data/prod5/simtel/${folder}/"
  output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/dl1/lstchain_v.${lstchain_version}/${folder}/"
  production_name="Prod5_${folder}_with_lstchain_v.${lstchain_version}"

  if [[ "${folder}" == "tag_nominal_LST_09_2020_v2" ]]; then
    json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/pmt_cam.json"
  else
    json_config_file="/fefs/home/david.miranda/software/cta-lstchain_v.${lstchain_version}/lstchain/data/sipm_cam.json"
  fi

  if [[ "${cluster}" == "camk" ]]; then
    echo "Not implemented yet"
    working_dir=" "
  elif [[ "${cluster}" == "yggdrasil" ]]; then
    echo "Not implemented yet"
    working_dir=" "
  elif [[ "${cluster}" == "itcluster" ]]; then
    working_dir="${path_to_lstchain}/lstchain/standard_analysis/scripts/mc_r0_to_dl1/"
  else
    echo "unknown server, exiting"
    exit 1
  fi

  bash ${working_dir}/mc_r0_to_dl1.sh "${input_folder}" "${output_folder}" "${json_config_file}" "${production_name}" "${working_dir}" "${cluster}"

  echo ${input_folder}
  echo ${output_folder}
  echo ${json_config_file}
  echo ${production_name}
  echo ${working_dir}
  echo ${cluster}

done
