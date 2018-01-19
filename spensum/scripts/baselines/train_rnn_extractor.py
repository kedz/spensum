import sys
import os
import argparse
import random
from collections import defaultdict

import spensum
import ntp
import torch
import rouge_papier
import pandas as pd


def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir):

    model.eval()

    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            texts = model.extract(batch.inputs, batch.metadata, 
                                  strategy="rank")

            for id, summary in zip(batch.metadata.id, texts):
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(config_path, max_ngram=2, lcs=False)
        return df[-1:]

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-labels", type=str, required=True)

    parser.add_argument("--train-summary-dir", type=str, required=True)
    parser.add_argument("--valid-summary-dir", type=str, required=True)

    #parser.add_argument("--valid", type=str, required=True)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=20, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--lr", required=False, default=.0001, type=float)
    parser.add_argument(
        "--batch-size", default=2, type=int, required=False)

    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)
    parser.add_argument(
        "--rnn-hidden-size", type=int, required=False, default=512)
    parser.add_argument(
        "--rnn-layers", type=int, required=False, default=1)



    parser.add_argument(
        "--hidden-layer-sizes", nargs="+", default=[100], type=int,
        required=False)
    parser.add_argument(
        "--hidden-layer-activations", nargs="+", default="relu", type=str,
        required=False)
    parser.add_argument(
        "--hidden-layer-dropout", default=.0, type=float, required=False)
    parser.add_argument(
        "--input-layer-norm", default=False, action="store_true")

    parser.add_argument(
        "--save-model", required=False, type=str)
#    parser.add_argument(
#        "--save-predictor", default=None, required=False, type=str)

    args = parser.parse_args(args)

    ntp.set_random_seed(args.seed)

    input_reader = spensum.dataio.init_duc_sds_input_reader(
        args.embedding_size)
    label_reader = spensum.dataio.init_duc_sds_label_reader()

    train_dataset = spensum.dataio.read_input_label_dataset(
        args.train_inputs, args.train_labels, 
        input_reader, label_reader,
        batch_size=args.batch_size, gpu=args.gpu)

    valid_dataset = spensum.dataio.read_input_label_dataset(
        args.valid_inputs, args.valid_labels, 
        input_reader, label_reader,
        batch_size=args.batch_size, gpu=args.gpu)

    model = spensum.model.RNNExtractor(
        args.embedding_size, args.rnn_hidden_size, layers=args.rnn_layers)
    if args.gpu > -1:
        model.cuda(args.gpu)

    non_salient_count = train_dataset.targets.eq(0).sum()
    salient_count = train_dataset.targets.eq(1).sum()
    weight = torch.FloatTensor([1 / non_salient_count, 1 / salient_count])
#    print("Training data:")
#    print("# salient = {}".format(salient_count))
#    print("# non-salient = {}".format(non_salient_count))

    opt = ntp.optimizer.Adam(model.parameters(), lr=args.lr)
    crit = ntp.criterion.BinaryCrossEntropy(
        mode="prob", weight=weight, mask_value=-1)
    crit.add_reporter(
        ntp.criterion.BinaryFMeasureReporter(mode="prob"))
    crit.set_selection_criterion("BinaryFMeasureReporter")

    #ntp.trainer.optimize_criterion(crit, model, opt, train_dataset,
    #                               validation_data=valid_dataset,
    #                               max_epochs=15)
    
    train_rouge_results = []
    valid_rouge_results = []
    best_rouge = 0

    for epoch in range(1, args.epochs + 1):


        def train_step_callback(step, max_steps, batch_loss, criterion):
            sys.stdout.write("\r")
            sys.stdout.write(" " * 79)
            sys.stdout.write("\r")
            sys.stdout.write(
                "\ttrain {}: {} / {} | obj: {:0.9f}".format(
                    epoch, step, max_steps, criterion.avg_loss))
            sys.stdout.flush()
            if step == max_steps:
                sys.stdout.write("\r" + " " * 79 + "\r")
                sys.stdout.flush()

        def valid_step_callback(step, max_steps, batch_loss, criterion):
            sys.stdout.write("\r")
            sys.stdout.write(" " * 79)
            sys.stdout.write("\r")
            sys.stdout.write(
                "\tvalid {}: {} / {} | obj: {:0.9f}".format(
                    epoch, step, max_steps, criterion.avg_loss))
            sys.stdout.flush()
            if step == max_steps:
                sys.stdout.write("\r" + " " * 79 + "\r")
                sys.stdout.flush()


        ntp.trainer.train_epoch(
            crit, model, opt, train_dataset, step_callback=train_step_callback)
        crit.checkpoint("training")

        #print(crit.report(indent="     "))
        #print(compute_rouge(model, train_dataset, args.train_summary_dir))

        #print("\n  * == Validation ==")

        ntp.trainer.eval(
            crit, model, valid_dataset, step_callback=valid_step_callback)
        crit.checkpoint("validation")
        

        #best_epoch, obj = crit.find_best_checkpoint("validation")
        #if best_epoch == epoch and save_model is not None:
        #    torch.save(model, save_model)
        #print(crit.report(indent="     "))
        #print("\n     Best epoch: {} obj: {}\n".format(
        #    best_epoch, obj))
        #print("")

        valid_rouge = compute_rouge(
            model, valid_dataset, args.valid_summary_dir)
        valid_rouge_results.append(valid_rouge)

        rouge_score = valid_rouge["rouge-2"].values[0]
        if rouge_score > best_rouge:
            best_rouge = rouge_score
            if args.save_model is not None:
                print("Saving model!")
                torch.save(model, args.save_model)


    #,
     #                              save_model=module_save_path)
    
    return pd.concat(valid_rouge_results, axis=0) 
    
    
    #exit()

    #train_data, valid_data = spensum.dataio.read_train_and_validation_data(
    #    args.train, args.valid, file_reader, args.batch_size, gpu=args.gpu)

if __name__ == "__main__": 
    main()
