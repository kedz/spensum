SPENSUM_DATA=${SPENSUM_DATA:="~/spensum_data"}
SPENSUM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SUMMARIES_DIR=$SPENSUM_DATA/duc-sds/summaries

#python $SPENSUM_DIR/python_main/preprocess/extract_duc_sds.py

parts=( "train" "valid" )
array=( "L" "1" "2")
for part in "${parts[@]}"
do
 for R in "${array[@]}"
 do
  python $SPENSUM_DIR/python_main/predictions_to_summary.py \
    -i $SPENSUM_DATA/duc-sds/inputs/duc.sds.inputs.${part}.json \
    -p $SPENSUM_DATA/duc-sds/ranks/duc.sds.ranks.seq.rouge-${R}.${part}.json \
    --output-dir $SUMMARIES_DIR/${part}/oracle.ranks.seq.rouge-${R} \
    -m ranks

  python $SPENSUM_DIR/python_main/predictions_to_summary.py \
    -i $SPENSUM_DATA/duc-sds/inputs/duc.sds.inputs.${part}.json \
    -p $SPENSUM_DATA/duc-sds/labels/duc.sds.labels.seq.rouge-${R}.${part}.json \
    --output-dir $SUMMARIES_DIR/${part}/oracle.labels.seq.rouge-${R} \
    -m labels

 done

 python ${SPENSUM_DIR}/python_main/evaluate_rouge.py \
    --system-summaries \
        $SUMMARIES_DIR/${part}/oracle.labels.seq.rouge-1 \
        $SUMMARIES_DIR/${part}/oracle.ranks.seq.rouge-1 \
        $SUMMARIES_DIR/${part}/oracle.labels.seq.rouge-2 \
        $SUMMARIES_DIR/${part}/oracle.ranks.seq.rouge-2 \
        $SUMMARIES_DIR/${part}/oracle.labels.seq.rouge-L \
        $SUMMARIES_DIR/${part}/oracle.ranks.seq.rouge-L \
    --system-names \
        oracle.seq.rouge-1.doc_order \
        oracle.seq.rouge-1.rank_order \
        oracle.seq.rouge-2.doc_order \
        oracle.seq.rouge-2.rank_order \
        oracle.seq.rouge-L.doc_order \
        oracle.seq.rouge-L.rank_order \
    --reference-summaries $SUMMARIES_DIR/${part}/human_abstracts \

done
exit





















RAW_DATA_DIR="${SPENSUM_DATA}/raw_data"
DUC_SDS_DATA_DIR="${SPENSUM_DATA}/duc_sds_data"
DUC_SDS_SUMMARY_DIR="${SPENSUM_DATA}/duc_sds_summaries"
DUC_SDS_LABEL_DIR="${SPENSUM_DATA}/duc_sds_labels"
ROUGE_PATH=${ROUGE_PATH:="~/tools/RELEASE-1.5.5"}

python ${SPENSUM_DIR}/python_main/setup_sds_data.py \
    --raw-data-dir $RAW_DATA_DIR \
    --output-data-dir $DUC_SDS_DATA_DIR \
    --summary-dir $DUC_SDS_SUMMARY_DIR \
    --extract-label-dir $DUC_SDS_LABEL_DIR

echo -e "DUC SDS TRAIN Label PRF\n"
python ${SPENSUM_DIR}/python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.train.tsv \
        $DUC_SDS_LABEL_DIR/lead3.train.tsv \
    --system-names rand lead3 \
    --reference-labels $DUC_SDS_LABEL_DIR/gold.train.tsv

echo -e "DUC SDS VALID Label PRF\n"
python ${SPENSUM_DIR}/python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.valid.tsv \
        $DUC_SDS_LABEL_DIR/lead3.valid.tsv \
    --system-names rand lead3 \
    --reference-labels $DUC_SDS_LABEL_DIR/gold.valid.tsv


echo -e "DUC SDS TEST Label PRF\n"
python ${SPENSUM_DIR}/python_main/evaluate_label_prf.py \
    --system-labels \
        $DUC_SDS_LABEL_DIR/rand.test.tsv \
        $DUC_SDS_LABEL_DIR/lead3.test.tsv \
    --system-names rand lead3 \
    --reference-labels $DUC_SDS_LABEL_DIR/gold.test.tsv


echo -e "DUC SDS TRAIN Rouge\n"
python ${SPENSUM_DIR}/python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/train/rand3 \
        $DUC_SDS_SUMMARY_DIR/train/lead3 \
        $DUC_SDS_SUMMARY_DIR/train/oracle \
    --system-names rand3 lead3 oracle \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/train/human_abstract \
    --rouge-dir $ROUGE_PATH

echo -e "DUC SDS VALID Rouge\n"
python ${SPENSUM_DIR}/python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/valid/rand3 \
        $DUC_SDS_SUMMARY_DIR/valid/lead3 \
        $DUC_SDS_SUMMARY_DIR/valid/oracle \
    --system-names rand3 lead3 oracle \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/valid/human_abstract \
    --rouge-dir $ROUGE_PATH

echo -e "DUC SDS TEST Rouge\n"
python ${SPENSUM_DIR}/python_main/evaluate_rouge.py \
    --system-summaries \
        $DUC_SDS_SUMMARY_DIR/test/rand3 \
        $DUC_SDS_SUMMARY_DIR/test/lead3 \
        $DUC_SDS_SUMMARY_DIR/test/oracle \
    --system-names rand3 lead3 oracle \
    --reference-summaries $DUC_SDS_SUMMARY_DIR/test/human_abstract \
    --rouge-dir $ROUGE_PATH
