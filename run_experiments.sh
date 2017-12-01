bash experiment_scripts/train_models.sh
bash experiment_scripts/predict_labels.sh
bash experiment_scripts/generate_summaries.sh
bash experiment_scripts/evaluate_labels.sh > results.txt
bash experiment_scripts/evaluate_summaries.sh >> results.txt

diff results.txt expected_results.txt
