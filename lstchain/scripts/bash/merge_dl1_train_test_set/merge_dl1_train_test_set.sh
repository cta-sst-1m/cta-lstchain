#!/bin/bash
#
# BASH script to launch jobs merging the dl1 data into 2 groups "train" and "test" with a small nuance,
# the RF models would be train with diffuse gammas and protons, and it will be tested in gamma point like and protons
# therefore, for training 100% gamma diffuse is used and 50% of protons, while for testing 100% gamma diffuse is used
# and 50% of protons
#
#
# input_folder :      path to the directory where the dl1 production is stored, i.e. the input directory
# output_folder :     path to the output directory
# percentage :        percentage of events to be trained, integer number from 1 to 100.
# production_name :   just a name label for display output
# working_dir :       path to the folder where the merge_dl1_train_test_set.sh file is located
# cluster :           name of the cluster that is used
#

input_folder=$1
output_folder=$2
percentage=$3
production_name=$4
working_dir=$5
cluster=$6

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  input_folder :         $input_folder "
echo "*-*-*-*-*-*-*-*-*-*  output_folder :        $output_folder "
echo "*-*-*-*-*-*-*-*-*-*  percentage :           $percentage "
echo "*-*-*-*-*-*-*-*-*-*  production_name :      $production_name "
echo "*-*-*-*-*-*-*-*-*-*  working_dir :          $working_dir "
echo "*-*-*-*-*-*-*-*-*-*  cluster :              $cluster "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

train_percent=${percentage}
test_percent=$((100-train_percent))
split_tag="${train_percent}.${test_percent}"

echo "run the merging for production $production_name"
for particle in "proton" "electron" "gamma_on" "gamma_off" "gamma_diffuse"; do

    set_dir="${output_folder}/${split_tag}/${particle}"
    mkdir -p ${set_dir}
    list_dl1_files="${set_dir}/list.txt"
    echo $list_dl1_files
    if [[ -f $list_dl1_files ]]; then
      rm $list_dl1_files
    fi
    ls $input_folder/$particle/*.h5 | grep -i -v merged |sort -V > $list_dl1_files

    if [[ "${particle}" == "gamma_on" ]]; then
      percentage=0
      n_dl1_files=$(cat $list_dl1_files | wc -l)
      n_dl1_train=$(( $n_dl1_files*$percentage/100 ))
    elif [[ "${particle}" == "gamma_off" ]]; then
      percentage=0
      n_dl1_files=$(cat $list_dl1_files | wc -l)
      n_dl1_train=$(( $n_dl1_files*$percentage/100 ))
    elif [[ "${particle}" == "gamma_diffuse" ]]; then
      percentage=100
      n_dl1_files=$(cat $list_dl1_files | wc -l)
      n_dl1_train=$(( $n_dl1_files*$percentage/100 ))
    else
      percentage=50
      n_dl1_files=$(cat $list_dl1_files | wc -l)
      n_dl1_train=$(( $n_dl1_files*$percentage/100 ))
    fi

    for run in "test" "train" ; do
        if [[ "$run" == "train" ]]; then
          if [[ $n_dl1_train -ne 0 ]]; then
            dl1_files_run=$(awk 'NR<='$n_dl1_train' {print}' $list_dl1_files)
          else
            continue
          fi
        elif [[ "$run" == "test" ]]; then
          if [[ $n_dl1_train -ne 1 ]]; then
            dl1_files_run=$(awk 'NR>'$n_dl1_train' {print}' $list_dl1_files)
          else
            continue
          fi
        else
            "WARNING: unknown run \"$run\" skipped"
            continue
        fi

        run_dir="${set_dir}/${run}"
        mkdir -p $run_dir

        nfile_run=$(echo $dl1_files_run| wc -w)
        for f in $dl1_files_run; do
            link_name="$run_dir/$(basename $f)"
            if [ -e "$link_name" ]; then
                echo "$link_name exists"
                continue
            fi
#            ln -t $run_dir -s $f; # when working with file
            ln -s $f $run_dir/$(basename $f); # when working with simlink of file (link of link)
        done
        merged_file="${output_folder}/${split_tag}/dl1_${particle}_merge_${run}.h5"
        if [ -e $merged_file ]; then
            echo "skiping the $run dataset with $particle for prod $prod as $merged_file exists"
            continue
        fi

        export merged_file=${merged_file}
        export run_dir=${run_dir}

        echo "submitting merging of $nfile_run dl1 files for the $run dataset with $particle for prod $production_name."
        echo "the output is in $merged_file"

        cd "${run_dir}"
        if [[ "${cluster}" == "camk" ]]; then
          echo "Not implemented yet"
          sleep 0.1
        elif [[ "${cluster}" == "yggdrasil" ]]; then
          echo "sbatch --export=merged_file=${merged_file},run_dir=${run_dir} ${working_dir}/${cluster}.sbatch"
          sbatch --export=merged_file=${merged_file},run_dir=${run_dir} ${working_dir}/${cluster}.sbatch
          sleep 0.2
        elif [[ "${cluster}" == "itcluster" ]]; then
          echo "sbatch --job-name=merge_dl1_${split_tag}_${particle} --output=merge_dl1_${split_tag}_${particle}.out --error=merge_dl1_${split_tag}_${particle}.out --export=merged_file=${merged_file},run_dir=${run_dir} ${working_dir}/${cluster}.sbatch"
          sbatch --job-name=merge_dl1_${split_tag}_${particle} --output=merge_dl1_${split_tag}_${particle}.out --error=merge_dl1_${split_tag}_${particle}.out --export=merged_file=${merged_file},run_dir=${run_dir} ${working_dir}/${cluster}.sbatch
          sleep 0.2
        else
          echo "unknown server, exiting"
          exit 1
        fi

    done
done
