#!/bin/bash

# all task
tasks=("statlog+australian+credit+approval" "statlog+german+credit+data" "ccFraud" "Credit_Card_Fraud_Detection" "Taiwan_Economic_Journal" "Travel_Insurance" "polish_companies_bankruptcy_data" "PortoSeguro" "lending_club")

for i in {1..3}
do
    for task in "${tasks[@]}"; do
        echo "Running experiment for task: $task"
        python run_experiment.py --exp_mode mcts --task "$task" --rollouts 10
        echo "Experiment for task $task completed."
    done
done

echo "[credit]All experiments completed."
