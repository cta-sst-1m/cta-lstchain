This is an example of runs you should make to get sensitivity results.
This is an example of the chain, you can adapt this scripts to loop over the data as percentage of events trained, type of productions, leakage maximum cut or intensity minimal cut.

The scripts listed below, loop over the values of the variables listed above. They call other bash scripts in the /lstchain/scripts/bash/* folders and are implemented with an structure to be run over different job manager systems, like slurm or sge (sge implementation is imminent).

You should run the scripts in the following order and wait for each step to conclude completely before running the next step.

1. mc_r0_to_dl1_run.sh

2. merge_dl1_train_test_set.sh

3. train_rf_scan.sh

4. get_rf_performance.sh

5. mc_dl1_to_dl2.sh

6. compute_sensitivity.sh

7. compare_sensitivity.sh

The last step "compare_sensitivity.sh" run for only one file, but its usage is to accept multiple inputs in order to compare with other sensitivities.