DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_MODEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_models"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"
DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"
DUC_SDS_LABEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_labels"
ROUGE_PATH="/home/kedz/projects2018/spensum/tools/RELEASE-1.5.5"


 ### Salience Module predictions ###

echo "Predicting labels with Salience Module on training dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-path $DUC_SDS_LABEL_DIR/salience.train.tsv

echo "Predicting labels with Salience Module on validation dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-path $DUC_SDS_LABEL_DIR/salience.valid.tsv

echo "Predicting labels with Salience Module on test dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-path $DUC_SDS_LABEL_DIR/salience.test.tsv


 #### Coverage Module predictions ###

echo "Predicting labels with Coverage Module on training dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/coverage.bin \
    --output-path $DUC_SDS_LABEL_DIR/coverage.train.tsv

echo "Predicting labels with Coverage Module on validation dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/coverage.bin \
    --output-path $DUC_SDS_LABEL_DIR/coverage.valid.tsv

echo "Predicting labels with Coverage Module on test dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/coverage.bin \
    --output-path $DUC_SDS_LABEL_DIR/coverage.test.tsv


 ### Novelty Module predictions ###

echo "Predicting labels with Novelty Module on training dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/novelty.bin \
    --output-path $DUC_SDS_LABEL_DIR/novelty.train.tsv

echo "Predicting labels with Novelty Module on validation dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/novelty.bin \
    --output-path $DUC_SDS_LABEL_DIR/novelty.valid.tsv

echo "Predicting labels with Novelty Module on test dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/novelty.bin \
    --output-path $DUC_SDS_LABEL_DIR/novelty.test.tsv

 ### Position Module predictions ###

echo "Predicting labels with Position Module on training dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/position.bin \
    --output-path $DUC_SDS_LABEL_DIR/position.train.tsv

echo "Predicting labels with Position Module on validation dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/position.bin \
    --output-path $DUC_SDS_LABEL_DIR/position.valid.tsv

echo "Predicting labels with Position Module on test dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/position.bin \
    --output-path $DUC_SDS_LABEL_DIR/position.test.tsv

echo "Predicting labels with Positional Salience Module on training dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/psalience.bin \
    --output-path $DUC_SDS_LABEL_DIR/psalience.train.tsv

echo "Predicting labels with Positional Salience Module on validation dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/psalience.bin \
    --output-path $DUC_SDS_LABEL_DIR/psalience.valid.tsv

echo "Predicting labels with Positional Salience Module on test dataset ..."
python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/psalience.bin \
    --output-path $DUC_SDS_LABEL_DIR/psalience.test.tsv
