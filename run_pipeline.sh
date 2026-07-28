#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00             # Adjust time as needed
#SBATCH --account=naiss2026-4-688-gpu

# --- UPDATE THESE TWO PATHS ---
PROJECT_DIR="/nobackup/proj/disk/av-ood-benchmarking/personal/seasonal-weather-ood/Weather_OOD_Analysis"
CONTAINER="/nobackup/proj/disk/av-ood-benchmarking/personal/seasonal-weather-ood/Weather_OOD_Analysis/bevformer_native.sif"

# 1. Define Master Variables
export OOD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export BASELINE_ID="20260502_220411"

WEATHERS=("Fog" "Snow")
SEVERITIES=("easy" "mid" "hard")

for CURRENT_WEATHER in "${WEATHERS[@]}"; do
  for CURRENT_SEVERITY in "${SEVERITIES[@]}"; do

    echo "========================================================="
    echo "STARTING FULL PIPELINE FOR: $CURRENT_WEATHER ($CURRENT_SEVERITY)"
    echo "========================================================="
    
    export OOD_WEATHER=$CURRENT_WEATHER
    export OOD_SEVERITY=$CURRENT_SEVERITY
    export OOD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # ---------------------------------------------------------
    # Step 1: Run BEVFormer Evaluation (4 GPUs)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 1: Model Evaluation..."
    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/data/sets/nuscenes-c/nuScenes-c:/workspace/data/nuScenes-c \
        --bind $PROJECT_DIR/data/sets/nuscenes:/workspace/data/nuscenes \
        --bind $PROJECT_DIR/data/sets/nuscenes:/workspace/nuscenes:ro \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --bind $PROJECT_DIR/core_models/BEVFormer/.dist_test:/workspace/core_models/BEVFormer/.dist_test:rw \
        --bind $PROJECT_DIR/evaluation_results:/workspace/test:rw \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$OOD_TIMESTAMP \
        $CONTAINER \
        torchrun --nproc_per_node=4 /workspace/core_models/BEVFormer/tools/test.py \
            /workspace/core_models/BEVFormer/projects/configs/bevformer/bevformer_base.py \
            /workspace/checkpoints/bevformer/bevformer_r101_dcn_24ep.pth \
            --launcher pytorch \
            --eval bbox \
            --tmpdir /tmp/bevformer_eval

    # ---------------------------------------------------------
    # Step 2: Calculate Energy Scores
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 2: Energy Scores..."

    LATEST_FOLDER=$(ls -td $PROJECT_DIR/data/intercepted_feature_logits/$OOD_WEATHER/$OOD_SEVERITY/*/ | head -1)
    ACTUAL_TIMESTAMP=$(basename $LATEST_FOLDER)

    echo "Found actual folder: $ACTUAL_TIMESTAMP"

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP \
        $CONTAINER \
        python safety_monitor/energy_score.py

    echo "Energy Score calculation complete!"

    # ---------------------------------------------------------
    # Step 3: Calculate Raw Mahalanobis Distance
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 3: (Raw) Mahalanobis Distance..."

    LATEST_FOLDER=$(ls -td $PROJECT_DIR/data/intercepted_feature_logits/$OOD_WEATHER/$OOD_SEVERITY/*/ | head -1)
    ACTUAL_TIMESTAMP=$(basename $LATEST_FOLDER)

    echo "Found actual folder: $ACTUAL_TIMESTAMP"

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID,NORMALIZATION=false \
        $CONTAINER \
        python safety_monitor/mahalanobis.py

    echo "Mahalanobis Distance (Raw) calculation complete!"

    # ---------------------------------------------------------
    # Step 4: Calculate Normalized Mahalanobis Distance
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 4: (Normalized) Mahalanobis Distance..."

    LATEST_FOLDER=$(ls -td $PROJECT_DIR/data/intercepted_feature_logits/$OOD_WEATHER/$OOD_SEVERITY/*/ | head -1)
    ACTUAL_TIMESTAMP=$(basename $LATEST_FOLDER)

    echo "Found actual folder: $ACTUAL_TIMESTAMP"

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID,NORMALIZATION=true \
        $CONTAINER \
        python safety_monitor/mahalanobis.py

    echo "Mahalanobis Distance (Normalized) calculation complete!"

    # ---------------------------------------------------------
    # Step 5: Evaluate AUROC (Raw)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 5: (Raw) AUROC Evaluation..."

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID,NORMALIZATION=false \
        $CONTAINER \
        python safety_monitor/auroc_evaluator.py

    echo "AUROC (Raw) plots generated!"

    # ---------------------------------------------------------
    # Step 6: Evaluate AUROC (Normalized)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 6: (Normalized) AUROC Evaluation..."

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID,NORMALIZATION=true \
        $CONTAINER \
        python safety_monitor/auroc_evaluator.py

    echo "AUROC (Normalized) plots generated!"

    # ---------------------------------------------------------
    # Step 7: Evaluate FPR95 (Raw)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 7: (Raw) FPR95 Evaluation..."

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID,NORMALIZATION=false \
        $CONTAINER \
        python safety_monitor/fpr95_evaluator.py

    echo "FPR95 (Raw) plots generated!"

    # ---------------------------------------------------------
    # Step 8: Evaluate FPR95 (Normalized)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 8: (Normalized) FPR95 Evaluation..."

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID,NORMALIZATION=true \
        $CONTAINER \
        python safety_monitor/fpr95_evaluator.py

    echo "FPR95 (Normalized) plots generated!"

    # ---------------------------------------------------------
    # Step 9: Evaluate Risk-Coverage Curve (Raw + Normalized)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 9: Risk-Coverage Curve..."

    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --bind $PROJECT_DIR/evaluation_results/:/workspace/core_models/BEVFormer/test \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$ACTUAL_TIMESTAMP,BASELINE_TIMESTAMP=$BASELINE_ID \
        $CONTAINER \
        python safety_monitor/risk_coverage_evaluator.py

    echo "Risk-Coverage Curve (Raw + Normalized) plots generated!"
    
    echo "========================================================="
    echo "PIPELINE COMPLETELY FINISHED!"
    echo "========================================================="
  done
done

echo "========================================================="
echo "ALL WEATHERS AND SEVERITIES COMPLETELY FINISHED!"
echo "========================================================="
