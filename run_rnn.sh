python python_main/train_rnn.py \
    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
    --train-labels /home/kedzie/spensum/datasets/nyt-sds/labels/nyt.sds.labels.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
    --valid-labels /home/kedzie/spensum/datasets/nyt-sds/labels/nyt.sds.labels.seq.rouge-1.sw.valid.json \
    --validation-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/valid/human_abstracts/ \
    --gpu 1 \
    --results-path /home/kedzie/spensum/results/nyt.sds.rnn.json \
    --model-path /home/kedzie/spensum/models/nyt.sds.rnn.bin \
    --epochs 50 \
    --context-dropout .7 \
    --batch-size 32 \
    --remove-stopwords


python python_main/train_rnn.py \
    --train-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.train.json \
    --train-labels /home/kedzie/spensum/datasets/duc-sds/labels/duc.sds.labels.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.valid.json \
    --valid-labels /home/kedzie/spensum/datasets/duc-sds/labels/duc.sds.labels.seq.rouge-1.sw.valid.json \
    --validation-summary-dir /home/kedzie/spensum/datasets/duc-sds/summaries/valid/human_abstracts/ \
    --gpu 1 \
    --results-path /home/kedzie/spensum/results/duc.sds.rnn.json \
    --model-path /home/kedzie/spensum/models/duc.sds.rnn.bin \
    --epochs 50 \
    --context-dropout .7 \
    #--context-size 200 \
    #--epochs 100

