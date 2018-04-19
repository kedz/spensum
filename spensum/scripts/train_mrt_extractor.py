import sys
import os
import argparse
import random
import math
from collections import defaultdict

import spensum
import ntp
import torch
import rouge_papier
import pandas as pd
import numpy as np
from spensum.model.mrt_extractor import MRTExtractor
from spensum.scripts.baselines.train_rnn_extractor import collect_reference_paths

def get_refs_dict():
  refs_paths = collect_reference_paths(args.train_summary_dir)
  refs_dict = dict()
  for id in refs_paths:
    refs_dict[id] = [line.strip().split(" ") for line in open(refs_paths[id]).readlines()]
  return refs_dict

def main(args=None):

  print("Running MRT training for query enchanced version")
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-inputs", type=str, required=True)
  parser.add_argument("--train-labels", type=str, required=True)
  parser.add_argument("--valid-inputs", type=str, required=True)
  parser.add_argument("--valid-labels", type=str, required=True)

  parser.add_argument("--train-summary-dir", type=str, required=True)
  parser.add_argument("--valid-summary-dir", type=str, required=True)

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

  args = parser.parse_args(args)

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

  refs_dict = get_refs_dict(args.train_summary_dir)
  
  model = MRTExtractor(args.embedding_size*2, args.rnn_hidden_size, layers=args.rnn_layers, refs_dict=refs_dict)
  if args.gpu > -1:
        model.cuda(args.gpu)

  optim = ntp.optimizer.Adam(model.parameters(), lr=args.lr)
  max_steps = math.ceil(train_dataset.size / train_dataset.batch_size)

  for epoch in range(1, args.epochs + 1):
    sys.stdout.write("epoch %s\n" % epoch)
    avg_expected_risks = []
    model.train()
    for step, batch in enumerate(train_dataset.iter_batch(), 1):
      optim.zero_grad()
      avg_expected_risk = model.forward(batch.inputs, batch.metadata)
      avg_expected_risks.append(avg_expected_risk.data[0])
      sys.stdout.write("{}/{} E[R]={:4.3f}\r".format(
        step, max_steps, np.mean(avg_expected_risks)))
      sys.stdout.flush()
      avg_expected_risk.backward()
      optim.step()

if __name__ == "__main__":
  main()
