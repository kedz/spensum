#python python_main/train_salience_predictions.py \
#    --train-salience /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.train.json \
#    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
#    --valid-salience /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.valid.json \
#    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
#    --gpu 0 \
#    --results-path /home/kedzie/spensum/results/nyt.sds.salience.json \
#    --model-path /home/kedzie/spensum/models/nyt.sds.salience.bin \
#    --dropout .55 \
#    --epochs 500
#
#exit

#python python_main/train_salience_predictions.py \
#    --train-salience /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.train.json \
#    --train-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.train.json \
#    --valid-salience /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.valid.json \
#    --valid-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.valid.json \
#    --gpu 0 \
#    --results-path /home/kedzie/spensum/results/duc.sds.salience.json \
#    --model-path /home/kedzie/spensum/models/duc.sds.salience.bin \
#    --dropout .55 \
#    --epochs 500

#exit
#
#
python python_main/predict_salience.py \
    --train-salience /home/kedzie/spensum/datasets/nyt-sds/pred-salience/nyt.sds.pred.salience.train.json \
    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
    --valid-salience /home/kedzie/spensum/datasets/nyt-sds/pred-salience/nyt.sds.pred.salience.valid.json \
    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
    --gpu 0 \
    --model-path /home/kedzie/spensum/models/nyt.sds.salience.bin \

#python python_main/predict_salience.py \
#    --train-salience /home/kedzie/spensum/datasets/duc-sds/pred-salience/duc.sds.pred.salience.train.json \
#    --train-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.train.json \
#    --valid-salience /home/kedzie/spensum/datasets/duc-sds/pred-salience/duc.sds.pred.salience.valid.json \
#    --valid-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.valid.json \
#    --gpu 0 \
#    --model-path /home/kedzie/spensum/models/duc.sds.salience.bin \


