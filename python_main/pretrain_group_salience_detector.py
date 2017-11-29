import argparse
import os

import ntp
import spen

import random 
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=100, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--lr", required=False, default=.01, type=float)
    parser.add_argument(
        "--batch-size", default=2, type=int, required=False)
    parser.add_argument(
        "--hidden-layer-dropout", default=.05, type=float, required=False)
    parser.add_argument(
        "--hidden-layer-sizes", nargs="+", default=[100], type=int, 
        required=False)
    parser.add_argument(
        "--hidden-layer-activations", nargs="+", default="relu", type=str,
        required=False)
    parser.add_argument(
        "--interaction-mode", default="concat", choices=["concat", "add"],
        type=str, required=False)
    
    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-predictor", default=None, required=False, type=str)
#    parser.add_argument(
#        "--save-results", default=None, required=False, type=str)

    args = parser.parse_args()        

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model = spen.BinaryGroupSalience(
        args.embedding_size,
        interaction_mode=args.interaction_mode,
        hidden_layer_sizes=args.hidden_layer_sizes,
        hidden_layer_activations=args.hidden_layer_activations,
        hidden_layer_dropout=args.hidden_layer_dropout)

    # Setup data fields and json reader.
    feature_field = ntp.dataio.field_reader.DenseVector(
        "embedding", 
        expected_size=args.embedding_size)
    label_field = ntp.dataio.field_reader.Label("label", vector_type=float)
    text_field = ntp.dataio.field_reader.String("text")
    fields = [feature_field, label_field, text_field]
    sequence_field = ntp.dataio.field_reader.Sequence(fields)
    file_reader = ntp.dataio.file_reader.JSONReader([sequence_field])

    # Read data
    file_reader.fit_parameters(args.data)
    (((features,), (labels,), (text,)), example_lengths), = file_reader.read(
        args.data)

    dataset = ntp.dataio.Dataset(
        (features, example_lengths, "inputs"),
        (labels, example_lengths, "targets"),
        (text, "text"),
        batch_size=args.batch_size, 
        shuffle=True, 
        gpu=-1,
        lengths=example_lengths)

    tr_idx, val_idx = ntp.trainer.generate_splits(
        [i for i in range(dataset.size)], valid_per=0)

    train_dataset = dataset.index_select(tr_idx)
    valid_dataset = dataset.index_select(val_idx)

    print(label_field.labels)
    non_salient_count = train_dataset.targets.eq(0).sum()
    salient_count = train_dataset.targets.eq(1).sum()
    weight = torch.FloatTensor([1 / non_salient_count, 1 / salient_count])

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
