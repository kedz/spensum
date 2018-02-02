
python python_main/train_oracle_pointer_network.py \
    --train-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.train.json \
    --train-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/nyt.sds.oracle.pn.json \
    --model-path /home/kedzie/spensum/models/nyt.sds.oracle.pn.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/valid/human_abstracts/ \
    --epochs 50 \
    --remove-stopwords \
    --lr .0001




python python_main/train_oracle_pointer_network.py \
    --train-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.train.json \
    --train-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/duc.sds.oracle.pn.json \
    --model-path /home/kedzie/spensum/models/duc.sds.oracle.pn.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/duc-sds/summaries/valid/human_abstracts/ \
    --epochs 50


