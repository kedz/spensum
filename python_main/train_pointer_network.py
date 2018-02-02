import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import ntp
import numpy as np
from spensum.model.sequence_standardizer import SequenceStandardizer
from spensum.model.pointer_network import PointerNetwork
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


def make_dataset(input_data, sal_data, tsne_data, rank_data, batch_size, gpu):
    
    lengths = torch.LongTensor([len(dp["inputs"]) for dp in input_data])
    max_length = lengths.max()
    dataset_size = len(tsne_data)
    sequence = torch.FloatTensor(dataset_size, max_length, 4).fill_(-1)
    ids = []
    texts = []
    ranks = []
    
    for i, (input_dp, sal_dp, tsne_dp, rank_dp) in enumerate(
                zip(input_data, sal_data, tsne_data, rank_data)):
        
        assert tsne_dp["id"] == rank_dp["id"]
        assert tsne_dp["id"] == input_dp["id"]
        assert tsne_dp["id"] == sal_dp["id"]

        ids.append(tsne_dp["id"])
        texts.append([s["text"] for s in input_dp["inputs"]])
        
        for j in range(len(rank_dp["ranks"])):      
            sequence[i,j,0] = sal_dp["salience"][j]      
            sequence[i,j,1] = input_dp["inputs"][j]["word_count"]
            sequence[i,j,2] = tsne_dp["tsne"][j][0]
            sequence[i,j,3] = tsne_dp["tsne"][j][1]
        #print(sequence[i,:len(rank_dp["ranks"])])
        ex_ranks = [(i, rank) for i, rank in enumerate(rank_dp["ranks"]) 
                    if rank > 0]
        ex_ranks.sort(key=lambda x: x[1])
        ranks.append([i for i, r in ex_ranks])
    
    target_lengths = torch.LongTensor([len(rank) for rank in ranks])
    
    max_rank_len = target_lengths.max()
    for rank in ranks:
        if len(rank) < max_rank_len:
            rank.extend([-1] * (max_rank_len - len(rank)))
    ranks = torch.LongTensor(ranks)

    layout = [
        ["inputs", [
            ["sequence", "sequence_inp"],
            ["length", "length_inp"]]
        ],
        ["targets", [
            ["sequence", "sequence_tgt"],
            ["length", "length_tgt"]]
        ],
        ["metadata", [
            ["id", "id"],
            ["text","text"]]
        ]
    ]

    dataset = ntp.dataio.Dataset(
        (sequence, lengths, "sequence_inp"),
        (lengths, "length_inp"),
        (ranks, target_lengths, "sequence_tgt"),
        (target_lengths, "length_tgt"),
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

def compute_rouge(model, dataset, reference_dir, remove_stopwords=True):

    model.eval()

    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            batch_size = batch.inputs.sequence.size(0)
            predictions = model.greedy_predict(batch.inputs)
            
            for b in range(batch_size):
                id = batch.metadata.id[b]
                preds = [p for p in predictions.data[b].cpu().tolist() 
                         if p > -1]
                summary = "\n".join([batch.metadata.text[b][p] for p in preds])                
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, 
            remove_stopwords=remove_stopwords)
        return df[-1:]

def train(opt, model, dataset, epoch, grad_clip=5):
    model.train()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        opt.zero_grad()
        logits = model(batch.inputs, batch.targets)
        targets = create_pointer_targets(batch)

        batch_size = logits.size(0)
        output_size = logits.size(1)

        logits_flat = logits.view(batch_size * output_size, -1)
        targets_flat = targets.view(-1)
        n_items = targets.gt(-1).float().data.sum()

        xent_pw = F.cross_entropy(
            logits_flat, targets_flat, size_average=False, 
            ignore_index=-1, reduce=False)
        avg_xent = xent_pw.sum() / n_items
        avg_xent.backward()
        total_xent += xent_pw.data.sum()
        total_els += n_items

        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)

        opt.step()
        sys.stdout.write(
            "train {}: {}/{} XENT={:0.6f}\r".format(
                epoch, n_iter, max_iters, total_xent / total_els))
        sys.stdout.flush()

    return total_xent / total_els


def validate(model, dataset, epoch, summary_dir, remove_stopwords=True):
    model.eval()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        logits = model(batch.inputs, batch.targets)
        targets = create_pointer_targets(batch)

        batch_size = logits.size(0)
        output_size = logits.size(1)

        logits_flat = logits.view(batch_size * output_size, -1)
        targets_flat = targets.view(-1)
        n_items = targets.gt(-1).float().data.sum()

        xent_pw = F.cross_entropy(
            logits_flat, targets_flat, size_average=False, 
            ignore_index=-1, reduce=False)
        avg_xent = xent_pw.sum() / n_items
        total_xent += xent_pw.data.sum()
        total_els += n_items

        sys.stdout.write("valid {}: {}/{} XENT={:0.6f}\r".format(
            epoch, n_iter, max_iters, total_xent / total_els))
        sys.stdout.flush()

    rouge_df = compute_rouge(
        model, dataset, summary_dir, remove_stopwords=remove_stopwords)
    r1, r2 = rouge_df.values[0].tolist()    
    
    return total_xent / total_els, r1, r2

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-salience", type=str, required=True)
    parser.add_argument("--train-tsne", type=str, required=True)
    parser.add_argument("--train-ranks", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-salience", type=str, required=True)
    parser.add_argument("--valid-tsne", type=str, required=True)
    parser.add_argument("--valid-ranks", type=str, required=True)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--lr", default=.001, type=float)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--context-dropout", default=.5, type=float)
    parser.add_argument("--context-size", default=300, type=int)
    parser.add_argument("--validation-summary-dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", default=48929234, type=int)

    args = parser.parse_args()

    ntp.set_random_seed(args.seed)

    embedding_size = 4

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
    training_inputs_data = read_data(args.train_inputs)
    print("Reading training salience data from {} ...".format(
        args.train_salience)) 
    training_salience_data = read_data(args.train_salience)
    print("Reading training tsne data from {} ...".format(
        args.train_tsne)) 
    training_tsne_data = read_data(args.train_tsne)
    print("Reading training rank data from {} ...".format(args.train_ranks)) 
    training_ranks_data = read_data(args.train_ranks)
    for a, b, c, d in zip(training_tsne_data, training_ranks_data, 
                          training_salience_data, training_inputs_data):
        assert a["id"] == b["id"]
        assert a["id"] == c["id"]
        assert a["id"] == d["id"]

    training_data = make_dataset(
        training_inputs_data, training_salience_data, training_tsne_data, 
        training_ranks_data, args.batch_size, args.gpu)
 
    print("Reading validation input data from {} ...".format(
        args.valid_inputs)) 
    validation_inputs_data = read_data(args.valid_inputs)
    print("Reading validation salience data from {} ...".format(
        args.valid_salience)) 
    validation_salience_data = read_data(args.valid_salience)
    print("Reading validation tsne data from {} ...".format(
        args.valid_tsne)) 
    validation_tsne_data = read_data(args.valid_tsne)
    print("Reading validation rank data from {} ...".format(args.valid_ranks)) 
    validation_ranks_data = read_data(args.valid_ranks)
    for a, b, c, d in zip(validation_tsne_data, validation_ranks_data,
                          validation_inputs_data, validation_salience_data):
        assert a["id"] == b["id"]
        assert a["id"] == c["id"]
        assert a["id"] == d["id"]

    validation_data = make_dataset(
        validation_inputs_data, validation_salience_data, 
        validation_tsne_data, validation_ranks_data, 
        args.batch_size, args.gpu)
    
    input_module = SequenceStandardizer(embedding_size)

    pn_model = PointerNetwork(
        input_module, args.context_size, attention_hidden_size=150, layers=2,
        context_dropout=args.context_dropout)

    for name, param in pn_model.named_parameters():
        if "weight" in name or name.startswith("W") or name == "v":
            nn.init.xavier_normal(param)    
        elif "bias" in name:
            nn.init.constant(param, 0)    
        else:
            nn.init.normal(param)

    if args.gpu > -1:
        pn_model.cuda(args.gpu)

    optim = torch.optim.Adam(pn_model.parameters(), lr=args.lr)

    train_xents = []
    valid_results = []

    best_rouge_2 = 0
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        
        train_xent = train(optim, pn_model, training_data, epoch)
        train_xents.append(train_xent)
        
        valid_result = validate(
            pn_model, validation_data, epoch, args.validation_summary_dir, 
            remove_stopwords=args.remove_stopwords)
        valid_results.append(valid_result)
        print("Epoch {} :: Train xent: {:0.3f} | Valid xent: {:0.3f} | R1: {:0.3f} | R2: {:0.3f}".format(
            epoch, train_xents[-1], *valid_results[-1]))
        if valid_results[-1][-1] > best_rouge_2:
            best_rouge_2 = valid_results[-1][-1]
            best_epoch = epoch
            if args.model_path is not None:
                print("Saving model ...")
                torch.save(pn_model, args.model_path)

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
        
main()

