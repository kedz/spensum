import argparse
import os

import ntp
import spen

import random 
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--valid", type=str, required=True)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=25, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--lr", required=False, default=.01, type=float)
    parser.add_argument(
        "--batch-size", default=8, type=int, required=False)
    parser.add_argument(
        "--num-positions", default=25, type=int, required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-predictor", default=None, required=False, type=str)

    args = parser.parse_args()        

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = spen.Position(args.num_positions)

    # Setup data fields and json reader.
    feature_field = ntp.dataio.field_reader.DenseVector(
        "embedding")
    label_field = ntp.dataio.field_reader.Label("label", vector_type=float)
    text_field = ntp.dataio.field_reader.String("text")
    fields = [feature_field, label_field, text_field]
    sequence_field = ntp.dataio.field_reader.Sequence(fields)
    file_reader = ntp.dataio.file_reader.JSONReader([sequence_field])

    # Read data
    file_reader.fit_parameters(args.train)
    tr_tensors, tr_example_lengths = file_reader.read(args.train)[0]
    (tr_features,), (tr_labels,), (tr_text,) = tr_tensors

    train_dataset = ntp.dataio.Dataset(
        (tr_features, tr_example_lengths, "inputs"),
        (tr_labels, tr_example_lengths, "targets"),
        (tr_text, "text"),
        batch_size=args.batch_size, 
        shuffle=True, 
        gpu=-1,
        lengths=tr_example_lengths)

    val_tensors, val_example_lengths = file_reader.read(args.valid)[0]
    (val_features,), (val_labels,), (val_text,) = val_tensors
    valid_dataset = ntp.dataio.Dataset(
        (val_features, val_example_lengths, "inputs"),
        (val_labels, val_example_lengths, "targets"),
        (val_text, "text"),
        batch_size=args.batch_size, 
        shuffle=True, 
        gpu=-1,
        lengths=val_example_lengths)

    non_salient_count = train_dataset.targets.eq(0).sum()
    salient_count = train_dataset.targets.eq(1).sum()
    weight = torch.FloatTensor([1 / non_salient_count, 1 / salient_count])
    print("Training data:")
    print("# salient = {}".format(salient_count))
    print("# non-salient = {}".format(non_salient_count))

    opt = ntp.optimizer.Adam(model.parameters(), lr=args.lr)
    crit = ntp.criterion.BinaryCrossEntropy(
        mode="prob", weight=weight, mask_value=-1)
    crit.add_reporter(
        ntp.criterion.BinaryFMeasureReporter(mode="prob"))
    crit.set_selection_criterion("BinaryFMeasureReporter")

    ntp.trainer.optimize_criterion(crit, model, opt, train_dataset,
                                   validation_data=valid_dataset,
                                   max_epochs=args.epochs,
                                   save_model=args.save_model)

    if args.save_predictor is not None:
        if args.save_model is None:
            raise Exception("Have to save model to save a predictor.")
        pred_dir = os.path.dirname(args.save_predictor)
        if pred_dir != "" and not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        print("Reading best model...")
        model = torch.load(args.save_model)
        data = {"model": model, 
                "feature_field": feature_field,
                "label_field": label_field, 
                "text_field": text_field}
        print("Saving model and readers...")
        torch.save(data, args.save_predictor)
    

    
if __name__ == "__main__":
    main()
