#!/bin/bash
# Era-restricted rerun driver: dynamic LR/XGB (48h main + 72h) + bootstrap,
# then GRU MPS seeds 42/43/44. Protocol/seeds/hyperparameters unchanged.
set -e
cd /Users/Yepsun/Mywork/Vscodeprojects/mimic
PY=/Users/Yepsun/myenv/bin/python3
E=results_vte/ajm/new/scripts_era
O=results_vte/ajm/new/output_era
D=results_vte/ajm/new/data_era

wait_npz() { # wait until npz exists AND is fully written (loadable)
  local f=$1
  until [ -f "$f" ] && $PY - "$f" <<'EOF' 2>/dev/null
import sys, numpy as np
z = np.load(sys.argv[1]); _ = [z[k].shape for k in z.files]
EOF
  do sleep 2; done
}

echo "=== START 47b-main $(date)"
$PY $E/47b_dynamic_models_v2.py \
  --train-view $D/landmark_features_train_inclprior_excl24h.parquet \
  --eval-view  $D/landmark_features_inclprior_excl24h.parquet \
  --out $O/dynamic_model_results_inclprior_excl24h.json &
PID47=$!
wait_npz $O/dynamic_oof_inputs_inclprior_excl24h.npz
echo "=== START bootstrap-main $(date)"
$PY $E/47b_bootstrap_only.py
wait $PID47
echo "=== DONE 47b-main $(date)"

echo "=== START 47b-72h $(date)"
$PY $E/47b_dynamic_models_72h.py \
  --train-view $D/landmark_features_train_72h_inclprior_excl24h.parquet \
  --eval-view  $D/landmark_features_72h_inclprior_excl24h.parquet \
  --out $O/dynamic_model_results_72h_inclprior_excl24h.json &
PID72=$!
wait_npz $O/dynamic_oof_inputs_72h_inclprior_excl24h.npz
echo "=== START bootstrap-72h $(date)"
$PY $E/47b_bootstrap_only_72h.py
wait $PID72
echo "=== DONE 47b-72h $(date)"

for S in 42 43 44; do
  echo "=== START gru-mps seed$S $(date)"
  $PY $E/48e_gru_mps_primary.py $S
  echo "=== DONE gru-mps seed$S $(date)"
done
echo "=== ALL DONE $(date)"
