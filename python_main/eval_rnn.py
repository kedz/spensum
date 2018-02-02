import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import ntp
import numpy as np
from spensum.model.sequence_standardizer import SequenceStandardizer
from spensum.model.rnn_model import RNNModel
from spensum.scripts.helper import create_pointer_targets
import rouge_papier
from collections import defaultdict


def read_data(inputs_path):
    dataset = []
    with open(inputs_path, "r") as fp:
        for line in fp:
            dataset.append(json.loads(line))
    return dataset

def get_summaries(id, summary_dir):
    filenames = [fn for fn in os.listdir(summary_dir) if fn.startswith(id)]
    texts = []
    for fn in filenames:
        path = os.path.join(summary_dir, fn)
        with open(path, "r") as fp:
            texts.append(fp.read())
    return texts


def make_dataset(input_data, batch_size, gpu):
    
    
    lengths = torch.LongTensor([len(example["inputs"]) 
                                for example in input_data])
    max_length = lengths.max()
    dataset_size = len(input_data)
    sequence = torch.FloatTensor(dataset_size, max_length, 300).fill_(-1)
    #targets = torch.LongTensor(dataset_size, max_length).fill_(-1)
    abs_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    rel_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    word_counts = torch.LongTensor(dataset_size, max_length).fill_(0)

    ids = []
    texts = []
    
    for i, input_dp in enumerate(input_data):
        
        ids.append(input_dp["id"])
        texts.append([sent["text"] for sent in input_dp["inputs"]])
        
        for j, sent in enumerate(input_dp["inputs"]):
            sequence[i,j].copy_(torch.FloatTensor(sent["embedding"]))
            abs_positions[i,j] = sent["sentence_id"]
            word_counts[i,j] = sent["word_count"]
        #targets[i,:lengths[i]].copy_(torch.LongTensor(label_dp["labels"]))
        
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
            ["word_count", "word_count"]]
        ],
        #["targets", "targets"],
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
        #(targets, lengths, "targets"),
        (ids, "id"),
        (texts, "text"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=lengths,
        layout=layout,
        shuffle=True)
    return dataset

def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir, remove_stopwords=True,
                  summary_length=100):

    model.eval()

    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            texts = model.predict(batch.inputs, batch.metadata)
            
            for b, text in enumerate(texts):
                id = batch.metadata.id[b]
                summary = "\n".join(text)                
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, 
            remove_stopwords=remove_stopwords,
            length=summary_length)
        return df[-1:]

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--test-inputs", type=str, required=True)

    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--summary-length", default=100, type=int)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-summary-dir", required=True, type=str)
    parser.add_argument("--valid-summary-dir", required=True, type=str)
    parser.add_argument("--test-summary-dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", default=48929234, type=int)

    args = parser.parse_args()

    ntp.set_random_seed(args.seed)

    if args.results_path is not None:
        results_dir = os.path.dirname(args.results_path)
        if not os.path.exists(results_dir) and results_dir != "":
            os.makedirs(results_dir)

    print("Loading model from {} ...".format(args.model_path))
    model = torch.load(args.model_path)
    if args.gpu > -1:
        model.cuda(args.gpu)
    else:
        model.cpu()

    print("Reading training input data from {} ...".format(
        args.train_inputs)) 
    training_input_data = read_data(args.train_inputs)

    training_data = make_dataset(
        training_input_data, args.batch_size, args.gpu)

    train_rouge_df = compute_rouge(
        model, training_data, args.train_summary_dir, 
        remove_stopwords=args.remove_stopwords,
        summary_length=args.summary_length)
    train_r1, train_r2 = train_rouge_df.values[0].tolist()    
    print("TRAIN R1 {:0.3f}  R2 {:0.3f}".format(train_r1, train_r2)) 

    print("Reading validation input data from {} ...".format(
        args.valid_inputs)) 
    validation_input_data = read_data(args.valid_inputs)

    validation_data = make_dataset(
        validation_input_data, args.batch_size, args.gpu)

    valid_rouge_df = compute_rouge(
        model, validation_data, args.valid_summary_dir, 
        remove_stopwords=args.remove_stopwords,
        summary_length=args.summary_length)
    valid_r1, valid_r2 = valid_rouge_df.values[0].tolist()    
    print("VALID R1 {:0.3f}  R2 {:0.3f}".format(valid_r1, valid_r2)) 

    print("Reading testing input data from {} ...".format(
        args.test_inputs)) 
    testing_input_data = read_data(args.test_inputs)

    testing_data = make_dataset(
        testing_input_data, args.batch_size, args.gpu)

    test_rouge_df = compute_rouge(
        model, testing_data, args.test_summary_dir, 
        remove_stopwords=args.remove_stopwords,
        summary_length=args.summary_length)
    test_r1, test_r2 = test_rouge_df.values[0].tolist()    
    print("TEST  R1 {:0.3f}  R2 {:0.3f}".format(test_r1, test_r2)) 

    results = {"training": {"rouge-1": train_r1, "rouge-2": train_r2},
               "validation": {"rouge-1": valid_r1, "rouge-2": valid_r2},
               "testing": {"rouge-1": test_r1, "rouge-2": test_r2}}

    with open(args.results_path, "w") as fp:
        fp.write(json.dumps(results))
  
if __name__ == "__main__":
    main()
