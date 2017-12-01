DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_MODEL_DIR="/home/kedz/projects2018/spensum/data/duc_sds_models"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"


 ### Word Count Module Pretrain ###

python python_main/pretrain/train_word_count_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-module $DUC_SDS_MODEL_DIR/word_count.bin \
    --save-predictor $DUC_SDS_PRED_DIR/word_count.bin

 ### Position Module Pretrain ###

python python_main/pretrain/train_position_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-module $DUC_SDS_MODEL_DIR/position.bin \
    --save-predictor $DUC_SDS_PRED_DIR/position.bin

 ### Salience Module Pretrain ###

python python_main/pretrain/train_salience_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-module $DUC_SDS_MODEL_DIR/salience.bin \
    --save-predictor $DUC_SDS_PRED_DIR/salience.bin

 ### Positional Salience Module Pretrain ###

python python_main/pretrain/train_positional_salience_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-module $DUC_SDS_MODEL_DIR/psalience.bin \
    --save-predictor $DUC_SDS_PRED_DIR/psalience.bin 

 ### Coverage Module Pretrain ###

python python_main/pretrain/train_coverage_module.py \
    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --save-module $DUC_SDS_MODEL_DIR/coverage.bin \
    --save-predictor $DUC_SDS_PRED_DIR/coverage.bin

# ### Novelty Module Pretrain ###
#
#python python_main/pretrain_novelty_module.py \
#    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
#    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
#    --salience-module $DUC_SDS_MODEL_DIR/salience.bin \
#    --save-model $DUC_SDS_MODEL_DIR/novelty.bin \
#    --save-predictor $DUC_SDS_PRED_DIR/novelty.bin
#
#

# ### Joint Module Pretrain ###
#
#python python_main/pretrain_joint_model.py \
#    --train $DUC_SDS_DATA_DIR/duc.sds.train.json \
#    --valid $DUC_SDS_DATA_DIR/duc.sds.valid.json \
#    --salience-module $DUC_SDS_MODEL_DIR/salience.bin \
#    --coverage-module $DUC_SDS_MODEL_DIR/coverage.bin \
#    --novelty-module $DUC_SDS_MODEL_DIR/novelty.bin \
#    --save-model $DUC_SDS_MODEL_DIR/joint.bin \
#    --save-predictor $DUC_SDS_PRED_DIR/joint.bin
