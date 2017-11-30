DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_MODEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_models"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"
DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"
DUC_SDS_LABEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_labels"
ROUGE_PATH="/home/kedz/projects2018/spensum/tools/RELEASE-1.5.5"

echo -e "DUC SDS TRAIN Rouge\n"
python python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/train/rand3 \
        $DUC_SDS_SUMMARY_DIR/train/lead3 \
        $DUC_SDS_SUMMARY_DIR/train/salience \
        $DUC_SDS_SUMMARY_DIR/train/coverage \
        $DUC_SDS_SUMMARY_DIR/train/novelty \
        $DUC_SDS_SUMMARY_DIR/train/position \
        $DUC_SDS_SUMMARY_DIR/train/oracle \
    --system-names rand3 lead3 salience coverage novelty position oracle\
    --reference-summaries $DUC_SDS_SUMMARY_DIR/train/human_abstract \
    --rouge-dir $ROUGE_PATH

echo -e "DUC SDS VALID Rouge\n"
python python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/valid/rand3 \
        $DUC_SDS_SUMMARY_DIR/valid/lead3 \
        $DUC_SDS_SUMMARY_DIR/valid/salience \
        $DUC_SDS_SUMMARY_DIR/valid/coverage \
        $DUC_SDS_SUMMARY_DIR/valid/novelty \
        $DUC_SDS_SUMMARY_DIR/valid/position \
        $DUC_SDS_SUMMARY_DIR/valid/oracle \
    --system-names rand3 lead3 salience coverage novelty position oracle \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/valid/human_abstract \
    --rouge-dir $ROUGE_PATH
