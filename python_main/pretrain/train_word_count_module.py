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
        "--epochs", default=15, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--lr", required=False, default=.1, type=float)
    parser.add_argument(
        "--batch-size", default=2, type=int, required=False)
    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)

    parser.add_argument(
        "--save-module", required=True, type=str)
    parser.add_argument(
        "--save-predictor", default=None, required=False, type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        torch.cuda.manual_seed(args.seed)

    module = spensum.module.WordCount(mode="pretrain")
    if args.gpu > -1:
        module.cuda(args.gpu)

    file_reader = spensum.dataio.initialize_sds_reader(args.embedding_size)

    train_data, valid_data = spensum.dataio.read_train_and_validation_data(
        args.train, args.valid, file_reader, args.batch_size, gpu=args.gpu)

    pretrained_module = spensum.trainer.pretrain_module(
        module, train_data, valid_data, args.lr, args.epochs, args.save_module)

    if args.save_predictor is not None:
        pred_dir = os.path.dirname(args.save_predictor)
        if pred_dir != "" and not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        data = {"model": pretrained_module, "file_reader": file_reader}
        print("Saving module and file reader...")
        torch.save(data, args.save_predictor)

if __name__ == "__main__":
    main()
