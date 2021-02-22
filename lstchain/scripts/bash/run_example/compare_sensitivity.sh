#!/bin/bash

# Choose sensitivity files path to compare different sensitivity productions at different cuts
tag_for_comparison=$1

lstchain_version="0.6.3"
path_to_lstchain="/home/david.miranda/software/cta-lstchain_v.${lstchain_version}"
rf_type="standard"

percent_event_train=(30 50)
ARRAY_INTENSITY=(100 200)
ARRAY_LEAKAGE=(0.0 0.2)

for percentage in "${percent_event_train[@]}"; do

  train_percent=${percentage}
  test_percent=$((100 - train_percent))
  split_tag="${train_percent}.${test_percent}"

  source /fefs/aswg/software/virtual_env/anaconda3/bin/activate lst-dev
  conda activate sipm_lstchain.0.6.3
  echo "conda activate sipm_lstchain.0.6.3"

  for leakage in "${ARRAY_LEAKAGE[@]}"; do
    for intensity in "${ARRAY_INTENSITY[@]}"; do
      # It compares different productions for the same leakage and intensity cut
      output_folder="/fefs/aswg/workspace/david.miranda/data/prod5/compare_sensitivity/lstchain_v.${lstchain_version}/${split_tag}/${rf_type}/${tag_for_comparison}/intensity_${intensity}_leakage_${leakage}/"

      # Define file path to compare
      f1=""
      f2=""
      ...
      fn=""

      # Define colors of productions to compare
      c1=""
      c2=""
      ...
      cn=""

      # Define markers of productions to compare
      m1=""
      m2=""
      ...
      mn=""

      # Define labels of productions to compare
      l1=""
      l2=""
      ...
      ln=""

      lstchain_unige_compare_sensitivity.py --input_files "$f1" "$f2" .. "$fn" --labels "$l1" "$l2" .. "$ln" --colors "$c1" "$c2" .. "$cn" --markers "$m1" "$m2" .. "$mn" --output_dir "${output_folder}"
    done
  done
done

# example :
# f1="${data_dir}/mono-lst-sipm-borofloat-3ns/${split_tag}/intensity_${intensity}_leakage_${leakage}/data/sensitivity.hdf5"
# f2="${data_dir}/mono-lst-sipm-borofloat-5ns/${split_tag}/intensity_${intensity}_leakage_${leakage}/data/sensitivity.hdf5"
# f3="${data_dir}/mono-lst-sipm-borofloat-7ns/${split_tag}/intensity_${intensity}_leakage_${leakage}/data/sensitivity.hdf5"
#
# l1="LST SiPM w/ filter 3ns"
# l2="LST SiPM w/ filter 5ns"
# l3="LST SiPM w/ filter 7ns"
#
# c1="tab:blue"
# c2="tab:orange"
# c3="tab:green"