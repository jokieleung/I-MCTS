# #!/bin/bash
tasks=("smoker-status" "software-defects" "jasmine" "credit-g" "Click_prediction_small" "kick" "kc1" "titanic" "icr" "wine-quality-white"  "mfeat-factors" "segment" "GesturePhaseSegmentationProcessed")

echo "[cls]starting experiments..."

for i in {1..3}
do
    for task in "${tasks[@]}"; do
        echo "Running experiment for task: $task"
        python run_experiment.py --exp_mode mcts --task "$task" --rollouts 10
        echo "Experiment for task $task completed."
    done
done

echo "[cls]All experiments completed."


