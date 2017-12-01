DUC_SDS_DATA_DIR="/home/kedz/projects2018/spensum/data/duc_sds_data"
DUC_SDS_PRED_DIR="/home/kedz/projects2018/spensum/data/duc_sds_predictors"
DUC_SDS_SUMMARY_DIR="/home/kedz/projects2018/spensum/data/duc_sds_summaries"


 ### Word Count Module generation ###

echo "Generating Word Count Module summaries on training dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/word_count.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/train/word_count 

echo "Generating Word Count Module summaries on validation dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/word_count.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/valid/word_count 

echo "Generating Word Count Module summaries on test dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/word_count.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/test/word_count 

 ### Position Module generation ###

echo "Generating Position Module summaries on training dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/position.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/train/position

echo "Generating Position Module summaries on validation dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/position.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/valid/position 

echo "Generating Position Module summaries on test dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/position.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/test/position 

 ### Salience Module generation ###

echo "Generating Salience Module summaries on training dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/train/salience 

echo "Generating Salience Module summaries on validation dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/valid/salience 

echo "Generating Salience Module summaries on test dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/salience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/test/salience 

 ### Coverage Module generation ###

echo "Generating Coverage Module summaries on training dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/coverage.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/train/coverage

echo "Generating Coverage Module summaries on validation dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/coverage.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/valid/coverage 

echo "Generating Coverage Module summaries on test dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/coverage.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/test/coverage 

 ### Positional Salience Module generation ###

echo "Generating Positional Salience Module summaries on training dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
    --predictor $DUC_SDS_PRED_DIR/psalience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/train/psalience

echo "Generating Positional Salience Module summaries on validation dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
    --predictor $DUC_SDS_PRED_DIR/psalience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/valid/psalience 

echo "Generating Positional Salience Module summaries on test dataset ..."
python python_main/generate_summaries.py \
    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
    --predictor $DUC_SDS_PRED_DIR/psalience.bin \
    --output-dir $DUC_SDS_SUMMARY_DIR/test/psalience 

#
# ### Novelty Module generation ###
#
#echo "Generating Novelty Module summaries on training dataset ..."
#python python_main/generate_summaries.py \
#    --data $DUC_SDS_DATA_DIR/duc.sds.train.json \
#    --predictor $DUC_SDS_PRED_DIR/novelty.bin \
#    --output-dir $DUC_SDS_SUMMARY_DIR/train/novelty
#
#echo "Generating Novelty Module summaries on validation dataset ..."
#python python_main/generate_summaries.py \
#    --data $DUC_SDS_DATA_DIR/duc.sds.valid.json \
#    --predictor $DUC_SDS_PRED_DIR/novelty.bin \
#    --output-dir $DUC_SDS_SUMMARY_DIR/valid/novelty 
#
#echo "Generating Novelty Module summaries on test dataset ..."
#python python_main/generate_summaries.py \
#    --data $DUC_SDS_DATA_DIR/duc.sds.test.json \
#    --predictor $DUC_SDS_PRED_DIR/novelty.bin \
#    --output-dir $DUC_SDS_SUMMARY_DIR/test/novelty 
#


