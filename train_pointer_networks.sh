python python_main/train_pointer_network.py \
    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
    --train-salience /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.train.json \
    --train-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.pred.train.json \
    --train-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
    --valid-salience /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.valid.json \
    --valid-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.pred.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/nyt.sds.pn.oracle.salience.pred.tsne.json \
    --model-path /home/kedzie/spensum/models/nyt.sds.pn.oracle.salience.pred.tsne.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/valid/human_abstracts/ \
    --epochs 50 \
    --remove-stopwords

python python_main/train_pointer_network.py \
    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
    --train-salience /home/kedzie/spensum/datasets/nyt-sds/pred-salience/nyt.sds.pred.salience.train.json \
    --train-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.train.json \
    --train-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
    --valid-salience /home/kedzie/spensum/datasets/nyt-sds/pred-salience/nyt.sds.pred.salience.valid.json \
    --valid-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/nyt.sds.pn.pred.salience.oracle.tsne.json \
    --model-path /home/kedzie/spensum/models/nyt.sds.pn.pred.salience.oracle.tsne.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/valid/human_abstracts/ \
    --epochs 50 \
    --remove-stopwords

python python_main/train_pointer_network.py \
    --train-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.train.json \
    --train-salience /home/kedzie/spensum/datasets/nyt-sds/pred-salience/nyt.sds.pred.salience.train.json \
    --train-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.pred.train.json \
    --train-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/nyt-sds/inputs/nyt.sds.inputs.valid.json \
    --valid-salience /home/kedzie/spensum/datasets/nyt-sds/pred-salience/nyt.sds.pred.salience.valid.json \
    --valid-tsne /home/kedzie/spensum/datasets/nyt-sds/tsne/nyt.sds.tsne.pred.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/nyt-sds/ranks/nyt.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/nyt.sds.pn.pred.salience.pred.tsne.json \
    --model-path /home/kedzie/spensum/models/nyt.sds.pn.pred.salience.pred.tsne.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/nyt-sds/summaries/valid/human_abstracts/ \
    --epochs 50 \
    --remove-stopwords


exit
python python_main/train_pointer_network.py \
    --train-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.train.json \
    --train-salience /home/kedzie/spensum/datasets/duc-sds/pred-salience/duc.sds.pred.salience.train.json \
    --train-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.train.json \
    --train-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.valid.json \
    --valid-salience /home/kedzie/spensum/datasets/duc-sds/pred-salience/duc.sds.pred.salience.valid.json \
    --valid-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 0 \
    --results-path /home/kedzie/spensum/results/duc.sds.pn.pred.salience.oracle.tsne.json \
    --model-path /home/kedzie/spensum/models/duc.sds.pn.pred.salience.oracle.tsne.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/duc-sds/summaries/valid/human_abstracts/ \
    --epochs 50


exit


python python_main/train_pointer_network.py \
    --train-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.train.json \
    --train-salience /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.train.json \
    --train-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.pred.train.json \
    --train-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.train.json \
    --valid-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.valid.json \
    --valid-salience /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.valid.json \
    --valid-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.pred.valid.json \
    --valid-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.valid.json \
    --gpu 1 \
    --results-path /home/kedzie/spensum/results/duc.sds.pn.oracle.salience.pred.tsne.json \
    --model-path /home/kedzie/spensum/models/duc.sds.pn.oracle.salience.pred.tsne.bin \
    --context-dropout .7 \
    --context-size 200 \
    --validation-summary-dir /home/kedzie/spensum/datasets/duc-sds/summaries/valid/human_abstracts/ \
    --epochs 50



#python python_main/train_pointer_network.py \
#    --train-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.train.json \
#    --train-salience /home/kedzie/spensum/datasets/duc-sds/pred-salience/duc.sds.pred.salience.train.json \
#    --train-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.pred.train.json \
#    --train-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.train.json \
#    --valid-inputs /home/kedzie/spensum/datasets/duc-sds/inputs/duc.sds.inputs.valid.json \
#    --valid-salience /home/kedzie/spensum/datasets/duc-sds/pred-salience/duc.sds.pred.salience.valid.json \
#    --valid-tsne /home/kedzie/spensum/datasets/duc-sds/tsne/duc.sds.tsne.pred.valid.json \
#    --valid-ranks /home/kedzie/spensum/datasets/duc-sds/ranks/duc.sds.ranks.seq.rouge-1.sw.valid.json \
#    --gpu 1 \
#    --results-path /home/kedzie/spensum/results/duc.sds.pn.pred.salience.pred.tsne.json \
#    --model-path /home/kedzie/spensum/models/duc.sds.pn.pred.salience.pred.tsne.bin \
#    --context-dropout .7 \
#    --context-size 200 \
#    --validation-summary-dir /home/kedzie/spensum/datasets/duc-sds/summaries/valid/human_abstracts/ \
#    --epochs 50


