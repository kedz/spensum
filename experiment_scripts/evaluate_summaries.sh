DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"
ROUGE_PATH="/home/kedz/projects2018/spensum/tools/RELEASE-1.5.5"


echo -e "DUC SDS TRAIN Rouge\n"
python python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/train/rand3 \
        $DUC_SDS_SUMMARY_DIR/train/lead3 \
        $DUC_SDS_SUMMARY_DIR/train/word_count \
        $DUC_SDS_SUMMARY_DIR/train/position \
        $DUC_SDS_SUMMARY_DIR/train/salience \
        $DUC_SDS_SUMMARY_DIR/train/coverage \
        $DUC_SDS_SUMMARY_DIR/train/psalience \
    --system-names \
        rand3 \
        lead3 \
        word_count \
        position \
        salience \
        coverage \
        psalience \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/train/human_abstract \
    --rouge-dir $ROUGE_PATH

echo -e "DUC SDS VALID Rouge\n"
python python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/valid/rand3 \
        $DUC_SDS_SUMMARY_DIR/valid/lead3 \
        $DUC_SDS_SUMMARY_DIR/valid/word_count \
        $DUC_SDS_SUMMARY_DIR/valid/position \
        $DUC_SDS_SUMMARY_DIR/valid/salience \
        $DUC_SDS_SUMMARY_DIR/valid/coverage \
        $DUC_SDS_SUMMARY_DIR/valid/psalience \
    --system-names \
        rand3 \
        lead3 \
        word_count \
        position \
        salience \
        coverage \
        psalience \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/valid/human_abstract \
    --rouge-dir $ROUGE_PATH
