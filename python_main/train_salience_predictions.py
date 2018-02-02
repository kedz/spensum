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

def get_summaries(id, summary_dir):
    filenames = [fn for fn in os.listdir(summary_dir) if fn.startswith(id)]
    texts = []
    for fn in filenames:
        path = os.path.join(summary_dir, fn)
        with open(path, "r") as fp:
            texts.append(fp.read())
    return texts

sw = set(["\"", "'", "''", "``", "`", ";", ",", ".", ":", "?", "!", "@", "-", "(", ")", "{", "}", "[", "]", "_", ". . ."])

def make_dataset(input_data, salience_data, batch_size, gpu):
    
    with open("/home/kedzie/spensum/datasets/nyt-sds/idfs.json", "r") as fp:
        idfs = json.loads(fp.read())

    lengths = torch.LongTensor([len(example["inputs"]) 
                                for example in input_data])
    max_length = lengths.max()
    dataset_size = len(input_data)
    sequence = torch.FloatTensor(dataset_size, max_length, 300).fill_(-1)
    targets = torch.FloatTensor(dataset_size, max_length).fill_(-1)
    abs_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    rel_positions = torch.LongTensor(dataset_size, max_length).fill_(0)
    word_counts = torch.LongTensor(dataset_size, max_length).fill_(0)
    mean_tfidfs = torch.LongTensor(dataset_size, max_length).fill_(0)

    ids = []
    texts = []
    
    for i, (input_dp, sal_dp) in enumerate(zip(input_data, salience_data)):
        
        assert input_dp["id"] == sal_dp["id"]

        ids.append(input_dp["id"])
        texts.append([sent["text"] for sent in input_dp["inputs"]])
        
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


        #print([(w,c) for w, c in sorted(tfidfs.items(), key=lambda x: x[1]) if w not in sw][-25:])

        #input()

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
        
        #input()

        targets[i,:lengths[i]].copy_(torch.FloatTensor(sal_dp["salience"]))
        
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
        (mean_tfidfs, lengths, "mean_tfidf"),
        (targets, lengths, "targets"),
        (ids, "id"),
        (texts, "text"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=lengths,
        layout=layout,
        shuffle=True)
    return dataset

def train(opt, model, dataset, epoch, grad_clip=5):
    model.train()
    total_loss = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        opt.zero_grad()
        salience = model(batch.inputs)

        mask = batch.targets.eq(-1)
        batch_size = salience.size(0)
        output_size = salience.size(1)
        num_els = batch.inputs.length.data.sum()

        loss = torch.norm(
            (salience - batch.targets).masked_fill_(mask, 0), 1)
        avg_loss = loss / num_els
        avg_loss.backward()

        total_loss += loss.data[0]
        total_els += num_els

        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)

        opt.step()
        sys.stdout.write(
            "train {}: {}/{} ERROR={:0.6f}\r".format(
                epoch, n_iter, max_iters, total_loss / total_els))
        sys.stdout.flush()

    return total_loss / total_els


def validate(model, dataset, epoch):
    model.eval()
    total_loss = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        salience = model(batch.inputs)

        mask = batch.targets.eq(-1)
        batch_size = salience.size(0)
        output_size = salience.size(1)
        num_els = batch.inputs.length.data.sum()

        loss = torch.norm(
            (salience - batch.targets).masked_fill_(mask, 0), 1)

        total_loss += loss.data[0]
        total_els += num_els

        sys.stdout.write(
            "train {}: {}/{} ERROR={:0.6f}\r".format(
                epoch, n_iter, max_iters, total_loss / total_els))
        sys.stdout.flush()

    return total_loss / total_els

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-salience", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-salience", type=str, required=True)
    parser.add_argument("--batch-size", default=300, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--lr", default=.00005, type=float)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dropout", default=.5, type=float)
    parser.add_argument("--context-size", default=300, type=int)
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

    print("Reading training salience data from {} ...".format(
        args.train_salience)) 
    training_salience_data = read_data(args.train_salience)

    print("Reading training inputs data from {} ...".format(args.train_inputs))
    training_inputs_data = read_data(args.train_inputs)
    for a, b in zip(training_inputs_data, training_salience_data):
        assert a["id"] == b["id"]

    training_data = make_dataset(
        training_inputs_data, training_salience_data, args.batch_size, 
        args.gpu)
 
    print("Reading validation salience data from {} ...".format(
        args.valid_salience)) 
    validation_salience_data = read_data(args.valid_salience)

    print("Reading validation inputs data from {} ...".format(args.valid_inputs)) 
    validation_inputs_data = read_data(args.valid_inputs)
    for a, b in zip(validation_salience_data, validation_inputs_data):
        assert a["id"] == b["id"]

    validation_data = make_dataset(
        validation_inputs_data, validation_salience_data, args.batch_size, 
        args.gpu)
    
    model = RougePredictor(dropout=args.dropout)
        #input_module, args.context_size, attention_hidden_size=150, layers=2,
        #context_dropout=args.context_dropout)

    for name, param in model.named_parameters():
        if "emb" in name:
            nn.init.normal(param)
        elif "weight" in name:
            nn.init.xavier_normal(param)    
        elif "bias" in name:
            nn.init.constant(param, 0)    
        else:
            nn.init.normal(param)

    if args.gpu > -1:
        model.cuda(args.gpu)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss = []
    valid_loss = []

    best_loss = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        
        train_loss.append(train(optim, model, training_data, epoch))
        
        valid_loss.append(validate(model, validation_data, epoch))
        print("Epoch {} :: Train err: {:0.5f} | Valid err: {:0.5f} | ".format(
            epoch, train_loss[-1], valid_loss[-1]))
        if valid_loss[-1] < best_loss:
            best_loss = valid_loss[-1]
            best_epoch = epoch
            if args.model_path is not None:
                print("Saving model ...")
                torch.save(model, args.model_path)

    print("Best epoch: {} Error={:0.5f}".format(
        best_epoch, valid_loss[best_epoch - 1]))

    if args.results_path is not None:
        results = {"training": train_loss,
                   "validation": valid_loss}
        print("Writing results to {} ...".format(args.results_path))
        with open(args.results_path, "w") as fp:
            fp.write(json.dumps(results)) 
        
if __name__ == "__main__":
    main()
