import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import ntp
import numpy as np
from spensum.model.rouge_predictor import RougePredictor


def read_data(inputs_path):
    dataset = []
    with open(inputs_path, "r") as fp:
        for line in fp:
            dataset.append(json.loads(line))
    return dataset

sw = set(["\"", "'", "''", "``", "`", ";", ",", ".", ":", "?", "!", "@", "-", "(", ")", "{", "}", "[", "]", "_", ". . ."])

def make_dataset(input_data, batch_size, gpu):

    with open("/home/kedzie/spensum/datasets/nyt-sds/idfs.json", "r") as fp:
        idfs = json.loads(fp.read())

    lengths = torch.LongTensor([len(example["inputs"]) 
                                for example in input_data])
    max_length = lengths.max()
    dataset_size = len(input_data)
    sequence = torch.FloatTensor(dataset_size, max_length, 300).fill_(-1)
    abs_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    rel_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    word_counts = torch.LongTensor(dataset_size, max_length).fill_(0)
    mean_tfidfs = torch.LongTensor(dataset_size, max_length).fill_(0)

    ids = []
    texts = []
    
    for i, input_dp in enumerate(input_data):
        
        num_words = 0
        tfidfs = {} 
        for sent in input_dp["inputs"]:
            #for word in sent.lower().split():
            for word in sent["tokens"]:
                word = word.lower()
                
                tfidfs[word] = tfidfs.get(word, 0) + 1
                num_words += 1
        for w in tfidfs.keys():
            tfidfs[w] = tfidfs[w] / num_words * idfs["idf"].get(w, np.log(idfs["num_docs"]))



        ids.append(input_dp["id"])
        texts.append([sent["text"] for sent in input_dp["inputs"]])
        
        avg_tfidfs = []
        for j, sent in enumerate(input_dp["inputs"]):
            sequence[i,j].copy_(torch.FloatTensor(sent["embedding"]))
            abs_positions[i,j] = sent["sentence_id"]
            word_counts[i,j] = sent["word_count"]
            avg_tfidfs.append(
                np.mean([tfidfs[w.lower()] for w in sent["tokens"] if w.lower() not in sw]))
        #print(avg_tfidfs)
        #print(np.arange(0,.055,.005))

        mean_tfidfs[i,:lengths[i]].copy_(
            torch.from_numpy(np.digitize(avg_tfidfs, np.arange(0,.055,.005))))
        
        pos_i = abs_positions[i,:lengths[i]].numpy()
        bins = np.linspace(1, pos_i[-1], 4, endpoint=False) 
        rel_pos_i = np.digitize(pos_i, bins)
        rel_positions[i,:lengths[i]].copy_(torch.from_numpy(rel_pos_i))

    layout = [
        ["inputs", [
            ["sequence", "sequence"],
            ["length", "length"],
            ["absolute_position", "absolute_position"],
            ["relative_position", "relative_position"],
            ["mean_tfidf", "mean_tfidf"],
            ["word_count", "word_count"]]
        ],
        ["metadata", [
            ["id", "id"],
            ["text","text"]]
        ]
    ]

    dataset = ntp.dataio.Dataset(
        (sequence, lengths, "sequence"),
        (lengths, "length"),
        (rel_positions, lengths, "relative_position"),
        (abs_positions, lengths, "absolute_position"),
        (word_counts, lengths, "word_count"),
        (mean_tfidfs, lengths, "mean_tfidf"),
        (ids, "id"),
        (texts, "text"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=lengths,
        layout=layout,
        shuffle=True)
    return dataset

def write_salience(model, dataset, path):
    
    with open(path, "w") as fp:
        for i in range(dataset.size):
            for batch in dataset[i].iter_batch():
                salience = model(batch.inputs).data[0].tolist()
                d = {"id": batch.metadata.id[0], "salience": salience}
                fp.write(json.dumps(d))
                fp.write("\n")

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--train-salience", type=str, default=None)
    parser.add_argument("--valid-salience", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", default=48929234, type=int)

    args = parser.parse_args()

    ntp.set_random_seed(args.seed)



    print("Reading model from {} ...".format(args.model_path))
    model = torch.load(args.model_path)

    if args.gpu > -1:
        model.cuda(args.gpu)
    else:
        model.cpu()

    if args.train_salience is not None:
        results_dir = os.path.dirname(args.train_salience)
        if not os.path.exists(results_dir) and results_dir != "":
            os.makedirs(results_dir)

    if args.valid_salience is not None:
        results_dir = os.path.dirname(args.valid_salience)
        if not os.path.exists(results_dir) and results_dir != "":
            os.makedirs(results_dir)

    print("Reading training inputs data from {} ...".format(args.train_inputs))
    training_data = make_dataset(
        read_data(args.train_inputs), args.batch_size, args.gpu)
    print("Writing training salience data to {} ...".format(
        args.train_salience))
    write_salience(model, training_data, args.train_salience)
 
    print("Reading validation inputs data from {} ...".format(
        args.valid_inputs)) 
    validation_data = make_dataset(
        read_data(args.valid_inputs), args.batch_size, args.gpu)
    print("Writing validation salience data to {} ...".format(
        args.valid_salience))
    write_salience(model, validation_data, args.valid_salience)
        
if __name__ == "__main__":
    main()
