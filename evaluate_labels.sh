DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_MODEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_models"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"
DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"
DUC_SDS_LABEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_labels"
ROUGE_PATH="/home/kedz/projects2018/spensum/tools/RELEASE-1.5.5"

echo -e "DUC SDS TRAIN Label PRF\n"
python python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.train.tsv \
        $DUC_SDS_LABEL_DIR/lead3.train.tsv \
        $DUC_SDS_LABEL_DIR/salience.train.tsv \
        $DUC_SDS_LABEL_DIR/coverage.train.tsv \
        $DUC_SDS_LABEL_DIR/novelty.train.tsv \
        $DUC_SDS_LABEL_DIR/position.train.tsv \
        $DUC_SDS_LABEL_DIR/psalience.train.tsv \
    --system-names rand lead3 salience coverage novelty position psalience \
    --reference-labels $DUC_SDS_LABEL_DIR/gold.train.tsv

echo -e "DUC SDS VALID Label PRF\n"
python python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.valid.tsv \
        $DUC_SDS_LABEL_DIR/lead3.valid.tsv \
        $DUC_SDS_LABEL_DIR/salience.valid.tsv \
        $DUC_SDS_LABEL_DIR/coverage.valid.tsv \
        $DUC_SDS_LABEL_DIR/novelty.valid.tsv \
        $DUC_SDS_LABEL_DIR/position.valid.tsv \
        $DUC_SDS_LABEL_DIR/psalience.valid.tsv \
    --system-names rand lead3 salience coverage novelty position psalience \
    --reference-labels $DUC_SDS_LABEL_DIR/gold.valid.tsv
