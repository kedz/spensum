python python_main/eval_rnn.py \
    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
    --train-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/train/human_abstracts/ \
    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
    --valid-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/valid/human_abstracts/ \
    --test-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.test.json \
    --test-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/test/human_abstracts/ \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/nyt.sds.summa.runner.eval.json \
    --model-path /home/kedzie/spensum/models/nyt.sds.summa.runner.bin \
    --epochs 50 \
    --batch-size 32 \
    --remove-stopwords


