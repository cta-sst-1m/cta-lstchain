#!/bin/bash
#
# BASH script that launches the job to compute the sensitivity for a given configuration and a given set of cuts
#
# dl2_gamma_test :        path to the dl2 gamma file
# dl2_proton_test         path to the dl2 proton file
# output_folder :         path to the folder where the sensitivity related data and plot would be stored
# intensity :             cut in the minimum intensity value. Intensity goes from 0 to infinity
# leakage :               cut in the maximum leakage value. Leakage goes from 0 to 1.
# gammanes_preselector :  value of the first gammaness used to discriminate protons and gammas (unoptimized value)
# max_gammaness :         maximum value to scan gammaness (standard use 0.8, however, why limit us to this value and not 1)
# magic_reference         path to the magic reference file
# production_name :       just a name label for display output
# working_dir :           path to the folder where the mc_dl1_to_dl2.sh file is located
# cluster :               which cluster is used, e.g. itcluster, camk, yggdrasil


dl2_gamma_test=$1
dl2_proton_test=$2
output_folder=$3
intensity=$4
leakage=$5
gammanes_preselector=$6
max_gammaness=$7
magic_reference=$8
production_name=$9
working_dir=${10}
cluster=${11}


echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*   dl2_gamma_test        :   ${dl2_gamma_test}"
echo "*-*-*-*-*-*-*-*-*-*   dl2_proton_test       :   ${dl2_proton_test}"
echo "*-*-*-*-*-*-*-*-*-*   output_folder         :   ${output_folder}"
echo "*-*-*-*-*-*-*-*-*-*   intensity             :   ${intensity}"
echo "*-*-*-*-*-*-*-*-*-*   leakage               :   ${leakage}"
echo "*-*-*-*-*-*-*-*-*-*   gammanes_preselector  :   ${gammanes_preselector}"
echo "*-*-*-*-*-*-*-*-*-*   max_gammaness         :   ${max_gammaness}"
echo "*-*-*-*-*-*-*-*-*-*   magic_reference       :   ${magic_reference}"
echo "*-*-*-*-*-*-*-*-*-*   production_name       :   ${production_name}"
echo "*-*-*-*-*-*-*-*-*-*   working_dir           :   ${working_dir}"
echo "*-*-*-*-*-*-*-*-*-*   cluster               :   ${cluster}"
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

mkdir -p "${output_folder}"

export dl2_gamma_test=${dl2_gamma_test}
export dl2_proton_test=${dl2_proton_test}
export output_folder=${output_folder}
export working_dir=${working_dir}
export cluster=${cluster}

export intensity=${intensity}
export leakage=${leakage}
export gammanes_preselector=${gammanes_preselector}
export max_gammaness=${max_gammaness}
export magic_reference=${magic_reference}

export production_name=${production_name}

cd ${output_folder}

if [[ "${cluster}" == "camk" ]]; then
  echo "Not implemented yet"
  sleep 0.2
elif [[ "${cluster}" == "yggdrasil" ]]; then
  echo sbatch --job-name=sensi_${intensity}_${leakage} --output=sensitivity.out --error=sensitivity.err --export=dl2_gamma_test=${dl2_gamma_test},dl2_proton_test=${dl2_proton_test},output_folder=${output_folder},working_dir=${working_dir},cluster=${cluster},intensity=${intensity},leakage=${leakage},max_gammaness=${max_gammaness},gammanes_preselector=${gammanes_preselector},magic_reference=${magic_reference} ${working_dir}/${cluster}.sbatch
  sbatch --job-name=sensi_${intensity}_${leakage} --output=sensitivity.out --error=sensitivity.err --export=dl2_gamma_test=${dl2_gamma_test},dl2_proton_test=${dl2_proton_test},output_folder=${output_folder},working_dir=${working_dir},cluster=${cluster},intensity=${intensity},leakage=${leakage},max_gammaness=${max_gammaness},gammanes_preselector=${gammanes_preselector},magic_reference=${magic_reference} ${working_dir}/${cluster}.sbatch
  sleep 0.2
elif [[ "${cluster}" == "itcluster" ]]; then
  echo sbatch --job-name=sensi_${intensity}_${leakage} --output=sensitivity.out --error=sensitivity.err --export=dl2_gamma_test=${dl2_gamma_test},dl2_proton_test=${dl2_proton_test},output_folder=${output_folder},working_dir=${working_dir},cluster=${cluster},intensity=${intensity},leakage=${leakage},max_gammaness=${max_gammaness},gammanes_preselector=${gammanes_preselector},magic_reference=${magic_reference} ${working_dir}/${cluster}.sbatch
  sbatch --job-name=sensi_${intensity}_${leakage} --output=sensitivity.out --error=sensitivity.err --export=dl2_gamma_test=${dl2_gamma_test},dl2_proton_test=${dl2_proton_test},output_folder=${output_folder},working_dir=${working_dir},cluster=${cluster},intensity=${intensity},leakage=${leakage},max_gammaness=${max_gammaness},gammanes_preselector=${gammanes_preselector},magic_reference=${magic_reference} ${working_dir}/${cluster}.sbatch
  sleep 0.2
else
  echo "unknown server, exiting"
  exit 1
fi


