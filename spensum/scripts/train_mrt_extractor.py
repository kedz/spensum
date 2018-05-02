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
from spensum.scripts.baselines.train_rnn_extractor import compute_rouge

def get_refs_dict(path, word_limit=100):
  refs_paths = collect_reference_paths(path)
  refs_dict = dict()
  for id in refs_paths:
    l = [line.strip().split(" ") for line in open(refs_paths[id][0]).readlines()]
    refs_dict[id] = [i for sl in l for i in sl][:word_limit]
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
        "--alpha", default=0.005, type=float)
  parser.add_argument(
        "--gamma", default=0.99, type=float)
  parser.add_argument(
        "--num-samples", default=5, type=int)
  parser.add_argument(
        "--budget", default=3, type=int)
  parser.add_argument(
        "--save-model", required=False, type=str)
  parser.add_argument(
        "--pretrained", required=True, type=str)
  parser.add_argument(
        "--stopwords", required=False, type=str, default="stopwords.txt")

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
  
  stopwords = set([word.strip() for word in open(args.stopwords).readlines()])

  try:
    model = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
    print("loaded pretrained model from %s successfully" % args.pretrained)
  except:
    model = MRTExtractor(args.embedding_size*2, args.rnn_hidden_size, 
                       layers=args.rnn_layers, refs_dict=refs_dict, 
                       budget=args.budget, num_samples=args.num_samples, 
                       alpha=args.alpha, gamma=args.gamma, stopwords=stopwords)
    print("failed to load pretrained model from %s, created a new model from scratch" % args.pretrained)

  if args.gpu > -1:
    model.cuda(args.gpu)

  optim = ntp.optimizer.Adam(model.parameters(), lr=args.lr)
  max_steps = math.ceil(train_dataset.size / train_dataset.batch_size)
  best_rouge = 0.0

  for epoch in range(1, args.epochs + 1):
    print("epoch %s" % epoch)
    
    # training
    avg_train_expected_risks = []
    model.train()
    for step, batch in enumerate(train_dataset.iter_batch(), 1):
      optim.zero_grad()
      avg_expected_risk = model.forward_mrt(batch.inputs, batch.metadata)
      avg_train_expected_risks.append(avg_expected_risk.data[0])
      avg_expected_risk.backward()
      optim.step()
    
    # evaluation
    avg_test_expected_risks = []
    model.eval()
    for step, batch in enumerate(valid_dataset.iter_batch(), 1):
      avg_expected_risk = model.forward_mrt(batch.inputs, batch.metadata)
      avg_test_expected_risks.append(avg_expected_risk.data[0])
   
    # get real rouge
    valid_rouge = compute_rouge(model, valid_dataset, args.valid_summary_dir)
    rouge_score1 = valid_rouge["rouge-1"].values[0]
    rouge_score2 = valid_rouge["rouge-2"].values[0]
    if rouge_score2 > best_rouge:
      best_rouge = rouge_score2
      if args.save_model is not None:
        print("Saving model!")
        torch.save(model, args.save_model)
 
    print("E[R]= train: %f, test: %f, rouge1=%f, rouge2=%f" % (np.mean(avg_train_expected_risks),np.mean(avg_test_expected_risks),rouge_score1,rouge_score2))
    sys.stdout.flush()
if __name__ == "__main__":
  main()
