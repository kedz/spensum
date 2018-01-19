import os
import sys
import argparse
import json
import re
import random

import torch
import rouge_papier
from duc_preprocess import duc2001
import ntp
from sklearn.decomposition import TruncatedSVD


def create_parent_dir(path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

def generate_extracts(data, mode, ngram, labels_path, ranks_path):

    create_parent_dir(labels_path)
    create_parent_dir(ranks_path)
    
    with open(labels_path, "w") as lbl_fp, open(ranks_path, "w") as rnk_fp:
        for input, target in data:
            
            input_text = [sent["text"] for sent in input]
            
            summaries = ["\n".join(sent["text"] for sent in sum["sentences"]) 
                         for sum in target]
            
            ranks, pairwise_ranks = rouge_papier.compute_extract(
                input_text, summaries, mode=mode, ngram=ngram, 
                remove_stopwords=True)
            labels = [min(r, 1) for r in ranks]
            id = "{}.{}".format(
                input[0]["docset_id"].lower(),
                input[0]["doc_id"].lower())

            lbl_fp.write(json.dumps({"id": id, "labels": labels}))
            lbl_fp.write("\n")
            rnk_fp.write(json.dumps({"id": id, "ranks": ranks, 
                                     "pairwise-ranks": pairwise_ranks}))
            rnk_fp.write("\n")



def generate_pairwise_ranks(data, mode, ngram):

    for input, target in data:

        input_text = [sent["text"] for sent in input]
        
        summaries = ["\n".join(sent["text"] for sent in sum["sentences"]) 
                     for sum in target]
        
        ranks = rouge_papier.compute_pairwise_ranks(
            input_text, summaries, mode=mode, ngram=ngram)


def write_summaries(data, output_dir):
    
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input, target in data:
        for summary in target:
            id = "{}.{}.{}.spl".format(
                summary["docset_id"].lower(), 
                summary["input_ids"][0].lower(),
                summary["summarizer"].lower())
            
            text = "\n".join([sent["text"] for sent in summary["sentences"]])
            
            path = os.path.join(output_dir, id)

            with open(path, "w") as fp:
                fp.write(text)

def generate_inputs(data, sif_embedding, inputs_path):

    create_parent_dir(inputs_path)

    with open(inputs_path, "w") as fp:
        for input, target in data:
            id = "{}.{}".format(
                input[0]["docset_id"].lower(),
                input[0]["doc_id"].lower())
            data = {"id": id, "inputs": []}
            input_sentences = []
            for i, sent in enumerate(input, 1):
                tokens = [token.lower() for token in sent["tokens"]]
                input_sentences.append(tokens)
                data_i = {"text": sent["text"], 
                          "sentence_id": i,
                          "word_count": len(sent["text"].split()),
                          "tokens": sent["tokens"]}
                data["inputs"].append(data_i)

            sent_embeddings = sif_embedding.embed_sentences(input_sentences)
            svd = TruncatedSVD(n_components=3, n_iter=50)
            svd.fit(sent_embeddings.numpy())
            data["principal_components"] = svd.components_.tolist()

            for i, sent_embed in enumerate(sent_embeddings):
                data["inputs"][i]["embedding"] = sent_embed.tolist()
            fp.write(json.dumps(data))
            fp.write("\n")

def generate_train_valid_splits(duc2001_path, output_root, valid_per=.15):
    
    orig_train_inputs = os.path.join(duc2001_path, "train", "inputs")
    orig_train_targets = os.path.join(duc2001_path, "train", "targets")
    orig_test_inputs = os.path.join(duc2001_path, "test", "inputs")
    orig_test_targets = os.path.join(duc2001_path, "test", "targets")
    total_examples = len(os.listdir(orig_train_inputs)) + \
        len(os.listdir(orig_test_inputs))

    train_and_valid_data = []
    for input_filename in os.listdir(orig_train_inputs):

        input_json = os.path.join(orig_train_inputs, input_filename)
        with open(input_json, "r") as inp_fp:
            input = json.loads(inp_fp.read())

        target_json = os.path.join(
            orig_train_targets, re.sub(r"input", r"target", input_filename))
        with open(target_json, "r") as tgt_fp:
            target = json.loads(tgt_fp.read())
        
        train_and_valid_data.append((input, target))

    for input_filename in os.listdir(orig_test_inputs):

        input_json = os.path.join(orig_test_inputs, input_filename)
        with open(input_json, "r") as inp_fp:
            input = json.loads(inp_fp.read())

        target_json = os.path.join(
            orig_test_targets, re.sub(r"input", r"target", input_filename))
        with open(target_json, "r") as tgt_fp:
            target = json.loads(tgt_fp.read())
        
        train_and_valid_data.append((input, target))

    # Shuffle train and valid data
    random.shuffle(train_and_valid_data)

    # Sort is stable so this will keep relatively shuffled but put 
    # inputs with multiple human references toward the end of the list.
    # We would prefer to have these in the validation set since they 
    # will give more reliable rouge scores.
    train_and_valid_data.sort(key=lambda x: len(x[1]))
    
    valid_size = int(total_examples * valid_per)
    train_data = train_and_valid_data[:-valid_size]
    valid_data = train_and_valid_data[-valid_size:]
    
    return train_data, valid_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spensum-data-path", type=str, default=None)
    parser.add_argument(
        "--duc-2001-data-path", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=43929524)

    args = parser.parse_args()

    ntp.set_random_seed(args.seed)

    if args.spensum_data_path is None:
        args.spensum_data_path  = os.getenv("SPENSUM_DATA", None)
        if args.spensum_data_path is None:
            sys.stderr.write(
                "Set SPENSUM_DATA to set location to write data.\n")
            sys.exit(1)

    if args.duc_2001_data_path is None:
        args.duc_2001_data_path  = os.getenv("DUC2001_ORIGINAL", None)
        if args.duc_2001_data_path is None:
            sys.stderr.write(
                "Set DUC2001_ORIGINAL to location of nist duc 2001 data.\n")
            sys.exit(1)

    pp_duc_2001_sds_path = os.path.join(
        args.spensum_data_path, "duc-sds", "preprocessed-data", "duc2001")

#    print("Preprocessing raw duc 2001 sds data ...")
#    duc2001.preprocess_sds(
#        pp_duc_2001_sds_path, nist_data_path=args.duc_2001_data_path, 
#        cnlp_port=9000)

    print("Loading sif embedding model ...")
    sif_emb = ntp.models.sentence_embedding.SIFEmbedding.from_pretrained()

    duc_sds_data_root = os.path.join(
        args.spensum_data_path, "duc-sds")

    train_data, valid_data = generate_train_valid_splits(
        pp_duc_2001_sds_path, duc_sds_data_root)

    all_training_sents = [[token.lower() for token in sent["tokens"]]
                          for ex in train_data
                          for sent in ex[0]]

    sif_emb.fit_principle_component(all_training_sents)
   
    sif_path = os.path.join(
        args.spensum_data_path, duc_sds_data_root, "sif.bin")
    torch.save(sif_emb, sif_path)

    print("Writing training inputs...")
    inputs_train_path = os.path.join(
        duc_sds_data_root, "inputs", "duc.sds.inputs.train.json")
    generate_inputs(train_data, sif_emb, inputs_train_path)

    print("Writing validation inputs...")
    inputs_valid_path = os.path.join(
        duc_sds_data_root, "inputs", "duc.sds.inputs.valid.json")
    generate_inputs(valid_data, sif_emb, inputs_valid_path)

#    print("Writing training summaries...")
#    summaries_train_path = os.path.join(
#        duc_sds_data_root, "summaries", "train", "human_abstracts")
#    write_summaries(train_data, summaries_train_path)
    
#    print("Writing validation summaries...")
#    summaries_valid_path = os.path.join(
#        duc_sds_data_root, "summaries", "valid", "human_abstracts")
#    write_summaries(valid_data, summaries_valid_path)

    #for mode in ["independent", "sequential"]:
    for mode in ["sequential"]:
        for part, data in [["valid", valid_data], ["train", train_data]]:
        #for mode in ["sequential"]:
            for rouge in [1]: #2, 3, 4]:

                labels_path = os.path.join(
                    duc_sds_data_root, "labels", 
                    "duc.sds.labels.{}.rouge-{}.sw.{}.json".format(
                        "indie" if mode == "independent" else "seq",
                        rouge, part))
                ranks_path = os.path.join(
                    duc_sds_data_root, "ranks", 
                    "duc.sds.ranks.{}.rouge-{}.sw.{}.json".format(
                        "indie" if mode == "independent" else "seq",
                        rouge, part))
                
                print("Generating {} rouge-{} ranks/labels " \
                      "for {} data".format(mode, rouge, part))

                generate_extracts(data, mode, rouge, labels_path, ranks_path)
                #generate_pairwise_ranks(data, mode, rouge)

if __name__ == "__main__":
    main()
