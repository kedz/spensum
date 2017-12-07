import os
import argparse
import random

import torch
import spensum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--valid", type=str, required=True)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=50, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--search-lr", required=False, default=.01, type=float)
    parser.add_argument(
        "--search-iters", required=False, default=50, type=int)

    parser.add_argument(
        "--lr", required=False, default=.01, type=float)
    parser.add_argument(
        "--batch-size", default=1, type=int, required=False)
    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)

    parser.add_argument(
        "--load-salience", type=str, required=False, default=None)
    parser.add_argument(
        "--salience", action="store_true", required=False, default=False)
    parser.add_argument(
        "--salience-hidden-layer-sizes", nargs="+", default=[100], type=int,
        required=False)
    parser.add_argument(
        "--salience-hidden-layer-activations", nargs="+", default="relu", 
        type=str, required=False)
    parser.add_argument(
        "--salience-hidden-layer-dropout", default=.25, type=float, 
        required=False)
    parser.add_argument(
        "--salience-input-layer-norm", default=False, action="store_true")

    parser.add_argument(
        "--load-position", type=str, required=False, default=None)
    parser.add_argument(
        "--position", action="store_true", required=False, default=False)
    parser.add_argument(
        "--position-num-positions", default=25, type=int, required=False)
    
    parser.add_argument(
        "--load-psalience", type=str, required=False, default=None)
    parser.add_argument(
        "--psalience", action="store_true", required=False, default=False)

    parser.add_argument(
        "--load-word-count", type=str, required=False, default=None)
    parser.add_argument(
        "--word-count", action="store_true", required=False, default=False)
    parser.add_argument(
        "--load-coverage", type=str, required=False, default=None)
    parser.add_argument(
        "--coverage", action="store_true", required=False, default=False)


#    parser.add_argument(
#        "--salience-module", type=str, required=True)
#    parser.add_argument(
#        "--coverage-module", type=str, required=True)
#    parser.add_argument(
#        "--novelty-module", type=str, required=True)
    
    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-predictor", default=None, required=False, type=str)

    args = parser.parse_args()        

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        torch.cuda.manual_seed(args.seed)

    modules = []

    if args.load_salience:
        print("Loading salience module from {} ...".format(args.load_salience))
        module = torch.load(
            args.load_salience, map_location=lambda storage, loc: storage)
        modules.append(module)
    elif args.salience:
        module = spensum.module.Salience(
            args.embedding_size,
            hidden_layer_sizes=args.salience_hidden_layer_sizes,
            hidden_layer_activations=args.salience_hidden_layer_activations,
            hidden_layer_dropout=args.salience_hidden_layer_dropout,
            input_layer_norm=args.salience_input_layer_norm,
            mode="pretrain")
        modules.append(module)

    if args.load_position:
        print("Loading position module from {} ...".format(args.load_position))
        module = torch.load(
            args.load_position, map_location=lambda storage, loc: storage)
        modules.append(module)
    elif args.position:
        module = spensum.module.Position(
            args.position_num_positions, 
            mode="pretrain")
        modules.append(module)

    if args.load_psalience:
        print("Loading positional salience module from {} ...".format(
            args.load_psalience))
        module = torch.load(
            args.load_psalience, map_location=lambda storage, loc: storage)
        modules.append(module)
    elif args.psalience:
        module = spensum.module.PositionalSalience(
            args.embedding_size,
            args.position_num_positions, 
            mode="pretrain")
        modules.append(module)

    if args.load_word_count:
        print("Loading word count module from {} ...".format(
            args.load_word_count))
        module = torch.load(
            args.load_word_count, map_location=lambda storage, loc: storage)
        modules.append(module)
    elif args.word_count:
        module = spensum.module.WordCount(mode="pretrain")
        modules.append(module)

    if args.load_coverage:
        print("Loading coverage module from {} ...".format(args.load_coverage))
        module = torch.load(
            args.load_coverage, map_location=lambda storage, loc: storage)
        modules.append(module)
    elif args.coverage:
        module = spensum.module.Coverage(
            args.embedding_size, 
            group_dropout=0,
            mode="pretrain")
        modules.append(module)

    if len(modules) == 0:
        raise Exception("Must have at least one module.")
    


    model = spensum.model.EnergyModel(
        modules, lr=args.search_lr, search_iters=args.search_iters)
    
    if args.gpu > -1:
        model.cuda(args.gpu)

    file_reader = spensum.dataio.initialize_sds_reader(args.embedding_size)

    train_data, valid_data = spensum.dataio.read_train_and_validation_data(
        args.train, args.valid, file_reader, args.batch_size, gpu=args.gpu)

    pretrained_model = spensum.trainer.train_model(
        model, train_data, valid_data, args.lr, args.epochs, args.save_model)

    if args.save_predictor is not None:
        pred_dir = os.path.dirname(args.save_predictor)
        if pred_dir != "" and not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        data = {"model": pretrained_model , "file_reader": file_reader}
        print("Saving module and file reader...")
        torch.save(data, args.save_predictor)
    
if __name__ == "__main__":
    main()
