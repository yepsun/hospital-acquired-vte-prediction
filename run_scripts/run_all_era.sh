#!/bin/bash
# Era-restricted (2008-2019) rerun of all MIMIC static-model analyses.
# Inputs: results_vte/ajm/new/data_era/ ; Outputs: results_vte/ajm/new/output_era/
set -e
cd /Users/Yepsun/Mywork/Vscodeprojects/mimic
E=results_vte/ajm/new/scripts_era
D=results_vte/ajm/new/data_era
O=results_vte/ajm/new/output_era
S98=results_vte/ajm/new/scripts/98_ajm_sensitivity_cohorts.py

run() { local tag=$1; shift; echo "=== START $tag $(date)"; python3 "$@"; echo "=== DONE $tag $(date)"; }

# 2x2 sensitivity matrix first (42b needs sensitivity_inclprior_excl24h_result.json)
run 98-ipe  $S98 --tag inclprior_excl24h --dataset $D/primary_inclprior_excl24h_model_dataset.parquet --splits $D/primary_inclprior_excl24h_splits.json --out $O/sensitivity_inclprior_excl24h_result.json
run 98-k24  $S98 --tag keep24h            --dataset $D/primary_keep24h_model_dataset.parquet            --splits $D/primary_keep24h_splits.json            --out $O/sensitivity_keep24h_result.json
run 98-e24  $S98 --tag excl24h            --dataset $D/sens_excl24h_model_dataset.parquet               --splits $D/sens_excl24h_splits.json               --out $O/sensitivity_excl24h_result.json
run 98-ip   $S98 --tag inclprior          --dataset $D/sens_inclprior_model_dataset.parquet             --splits $D/sens_inclprior_splits.json             --out $O/sensitivity_inclprior_result.json

run 39b $E/39b_baseline_scores_eval_v2.py
run 61  $E/61_static_oof_multiseed.py
run 41b $E/41b_calibration_dca_v2.py
run 42b $E/42b_temporal_subgroup_v2.py
run 60  $E/60_service_subgroups_v2.py
run 55b $E/55b_ed_vitals_sensitivity_v2.py
run 50c $E/50c_strict_control_apples2apples_v2.py
run 62  $E/62_review_gaps.py
run 57  $E/57_threshold_calibration_figures_v2.py

# dynamic eval view needed to regenerate the npz consumed by 47b_bootstrap_only
echo "=== START 47b-dynamic-pass1 $(date)"
python3 $E/47b_dynamic_models_v2.py \
  --train-view $D/landmark_features_train_inclprior_excl24h.parquet \
  --eval-view  $D/landmark_features_inclprior_excl24h.parquet \
  --out $O/dynamic_model_results_inclprior_excl24h.json || echo "pass1 ended (expected: inference json not yet present)"
run 47b-boot $E/47b_bootstrap_only.py
run 47b-dynamic-pass2 $E/47b_dynamic_models_v2.py \
  --train-view $D/landmark_features_train_inclprior_excl24h.parquet \
  --eval-view  $D/landmark_features_inclprior_excl24h.parquet \
  --out $O/dynamic_model_results_inclprior_excl24h.json

echo "=== ALL DONE $(date)"
