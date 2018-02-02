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


def make_dataset(input_data, label_data, batch_size, gpu):
    
    
    lengths = torch.LongTensor([len(example["inputs"]) 
                                for example in input_data])
    max_length = lengths.max()
    dataset_size = len(input_data)
    sequence = torch.FloatTensor(dataset_size, max_length, 300).fill_(-1)
    targets = torch.LongTensor(dataset_size, max_length).fill_(-1)
    abs_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    rel_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    word_counts = torch.LongTensor(dataset_size, max_length).fill_(0)

    ids = []
    texts = []
    
    for i, (input_dp, label_dp) in enumerate(zip(input_data, label_data)):
        
        assert input_dp["id"] == label_dp["id"]

        ids.append(input_dp["id"])
        texts.append([sent["text"] for sent in input_dp["inputs"]])
        
        for j, sent in enumerate(input_dp["inputs"]):
            sequence[i,j].copy_(torch.FloatTensor(sent["embedding"]))
            abs_positions[i,j] = sent["sentence_id"]
            word_counts[i,j] = sent["word_count"]
        targets[i,:lengths[i]].copy_(torch.LongTensor(label_dp["labels"]))
        
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
        ["targets", "targets"],
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
        (targets, lengths, "targets"),
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

def train(opt, model, dataset, epoch, grad_clip=5):
    model.train()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        opt.zero_grad()
        logits = model(batch.inputs)

        mask = batch.targets.gt(-1).float()
        n_items = mask.data.sum()

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            size_average=False)

        avg_loss = bce / n_items
        avg_loss.backward()

        total_xent += bce.data[0]
        total_els += n_items


        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)

        opt.step()
        sys.stdout.write(
            "train {}: {}/{} XENT={:0.6f}\r".format(
                epoch, n_iter, max_iters, total_xent / total_els))
        sys.stdout.flush()

    return total_xent / total_els

def validate(model, dataset, epoch, summary_dir, remove_stopwords=True,
             summary_length=100):
    model.eval()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        logits = model(batch.inputs)

        mask = batch.targets.gt(-1).float()
        n_items = mask.data.sum()

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            size_average=False)

        avg_loss = bce / n_items
        avg_loss.backward()

        total_xent += bce.data[0]
        total_els += n_items

        sys.stdout.write("valid {}: {}/{} XENT={:0.6f}\r".format(
            epoch, n_iter, max_iters, total_xent / total_els))
        sys.stdout.flush()

    rouge_df = compute_rouge(
        model, dataset, summary_dir, remove_stopwords=remove_stopwords,
        summary_length=summary_length)
    r1, r2 = rouge_df.values[0].tolist()    
    
    return total_xent / total_els, r1, r2

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-labels", type=str, required=True)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--summary-length", default=100, type=int)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--context-dropout", default=.5, type=float)
    parser.add_argument("--context-size", default=200, type=int)
    parser.add_argument("--validation-summary-dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", default=48929234, type=int)


    args = parser.parse_args()

    ntp.set_random_seed(args.seed)

    embedding_size = 300

    if args.results_path is not None:
        results_dir = os.path.dirname(args.results_path)
        if not os.path.exists(results_dir) and results_dir != "":
            os.makedirs(results_dir)

    if args.model_path is not None:
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir) and model_dir != "":
            os.makedirs(model_dir)

    print("Reading training input data from {} ...".format(
        args.train_inputs)) 
    training_input_data = read_data(args.train_inputs)

    print("Reading training label data from {} ...".format(args.train_labels)) 
    training_label_data = read_data(args.train_labels)
    for a, b in zip(training_input_data, training_label_data):
        assert a["id"] == b["id"]

    training_data = make_dataset(
        training_input_data, training_label_data, args.batch_size, args.gpu)
 
    print("Reading validation input data from {} ...".format(
        args.valid_inputs)) 
    validation_input_data = read_data(args.valid_inputs)

    print("Reading validation label data from {} ...".format(
        args.valid_labels)) 
    validation_label_data = read_data(args.valid_labels)
    for a, b in zip(validation_input_data, validation_label_data):
        assert a["id"] == b["id"]

    validation_data = make_dataset(
        validation_input_data, validation_label_data, args.batch_size, 
        args.gpu)
    
    model = RNNModel(hidden_size=args.context_size, 
        dropout=args.context_dropout)

    for name, param in model.named_parameters():
        if "emb" not in name and "weight" in name:
            nn.init.xavier_normal(param)    
        elif "emb" not in name and "bias" in name:
            nn.init.constant(param, 0)    

    if args.gpu > -1:
        model.cuda(args.gpu)

    optim = torch.optim.Adam(model.parameters(), lr=.001)

    train_xents = []
    valid_results = []

    best_rouge_2 = 0
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        
        train_xent = train(optim, model, training_data, epoch)
        train_xents.append(train_xent)
        
        valid_result = validate(
            model, validation_data, epoch, args.validation_summary_dir, 
            remove_stopwords=args.remove_stopwords, 
            summary_length=args.summary_length)
        valid_results.append(valid_result)
        print(("Epoch {} :: Train xent: {:0.3f} | Valid xent: {:0.3f} | " \
               "R1: {:0.3f} | R2: {:0.3f}").format(
                  epoch, train_xents[-1], *valid_results[-1]))

        if valid_results[-1][-1] > best_rouge_2:
            best_rouge_2 = valid_results[-1][-1]
            best_epoch = epoch
            if args.model_path is not None:
                print("Saving model ...")
                torch.save(model, args.model_path)

    print("Best epoch: {}  ROUGE-1 {:0.3f}  ROUGE-2 {:0.3f}".format(
        best_epoch, *valid_results[best_epoch - 1][1:]))
    
    if args.results_path is not None:
        results = {"training": {"cross-entropy": train_xents},
                   "validation": {
                       "cross-entropy": [x[0] for x in valid_results], 
                       "rouge-1": [x[1] for x in valid_results],
                       "rouge-2": [x[2] for x in valid_results]}}
        print("Writing results to {} ...".format(args.results_path))
        with open(args.results_path, "w") as fp:
            fp.write(json.dumps(results)) 
   
if __name__ == "__main__":
    main()

