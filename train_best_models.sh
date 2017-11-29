DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_MODEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_models"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"
DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"
DUC_SDS_LABEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_labels"
ROUGE_PATH="/home/kedz/projects2018/spensum/tools/RELEASE-1.5.5"

python python_main/pretrain_salience_detector.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-model $DUC_SDS_MODEL_DIR/salience.bin \
    --save-predictor $DUC_SDS_PRED_DIR/salience.bin


python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-path $DUC_SDS_LABEL_DIR/salience.train.tsv

python python_main/predict_labels.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-path $DUC_SDS_LABEL_DIR/salience.valid.tsv



echo -e "DUC SDS TRAIN Label PRF\n"
python python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.train.tsv \
        $DUC_SDS_LABEL_DIR/lead3.train.tsv \
        $DUC_SDS_LABEL_DIR/salience.train.tsv \
    --system-names rand lead3 salience\
    --reference-labels $DUC_SDS_LABEL_DIR/gold.train.tsv

echo -e "DUC SDS VALID Label PRF\n"
python python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.valid.tsv \
        $DUC_SDS_LABEL_DIR/lead3.valid.tsv \
        $DUC_SDS_LABEL_DIR/salience.valid.tsv \
    --system-names rand lead3 salience \
    --reference-labels $DUC_SDS_LABEL_DIR/gold.valid.tsv



python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/train/salience 

python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/valid/salience 

echo -e "DUC SDS TRAIN Rouge\n"
python python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/train/rand3 \
        $DUC_SDS_SUMMARY_DIR/train/lead3 \
        $DUC_SDS_SUMMARY_DIR/train/salience \
    --system-names rand3 lead3 salience \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/train/human_abstract \
    --rouge-dir $ROUGE_PATH

echo -e "DUC SDS VALID Rouge\n"
python python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/valid/rand3 \
        $DUC_SDS_SUMMARY_DIR/valid/lead3 \
        $DUC_SDS_SUMMARY_DIR/valid/salience \
    --system-names rand3 lead3 salience \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/valid/human_abstract \
    --rouge-dir $ROUGE_PATH


