DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_MODEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_models"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"
DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"
DUC_SDS_LABEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_labels"
ROUGE_PATH="/home/kedz/projects2018/spensum/tools/RELEASE-1.5.5"


 ### Salience Module Pretrain ###

python python_main/pretrain_salience_detector.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-model $DUC_SDS_MODEL_DIR/salience.bin \
    --save-predictor $DUC_SDS_PRED_DIR/salience.bin

 ### Coverage Module Pretrain ###

python python_main/pretrain_coverage_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-model $DUC_SDS_MODEL_DIR/coverage.bin \
    --save-predictor $DUC_SDS_PRED_DIR/coverage.bin

 ### Novelty Module Pretrain ###

python python_main/pretrain_novelty_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --salience-module $DUC_SDS_MODEL_DIR/salience.bin \
    --save-model $DUC_SDS_MODEL_DIR/novelty.bin \
    --save-predictor $DUC_SDS_PRED_DIR/novelty.bin
