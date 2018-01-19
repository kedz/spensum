import argparse
import os
import sys
import random
import ntp
import json
import rouge_papier
import multiprocessing as mp
from sklearn.decomposition import TruncatedSVD
import torch


def create_parent_dir(path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

def make_train_valid_list(data_dir):
    file_ids = sorted(os.listdir(data_dir))
    random.shuffle(file_ids)
    valid_size = int(len(file_ids) * .10)

    train_ids = file_ids[:-valid_size]
    valid_ids = file_ids[-valid_size:]

    return train_ids, valid_ids

def read_inputs(data_dir, file_ids=None):
    
    if file_ids is None:
        file_ids = sorted(os.listdir(data_dir))

    examples = []
    for i, file_id in enumerate(file_ids, 1):
        sys.stdout.write("{}/{}\r".format(i, len(file_ids)))
        sys.stdout.flush()

        path = os.path.join(data_dir, file_id)
        input_sentences = []
        with open(path, "r") as fp:
            fp.readline()

            current_sentence = []
            for line in fp:
                items = line.split("\t")
                if len(items) > 1:
                    current_sentence.append(items[3])
                else:
                    if len(current_sentence) > 0:
                        tokens = current_sentence
                        text = " ".join(current_sentence)
                        word_count = len(tokens)
                        sentence_id = len(input_sentences) + 1
                        input_sentences.append({
                            "text": text,
                            "tokens": tokens,
                            "sentence_id": sentence_id,
                            "word_count": word_count})
                    
                    current_sentence = []
                    
        examples.append({'id': file_id, 'inputs': input_sentences})
    print("")
        
    return examples

def generate_inputs(data, sif_embedding, inputs_path):

    create_parent_dir(inputs_path)

    with open(inputs_path, "w") as fp:
        for example in data:
            input_sentences = []
            for sent in example["inputs"]:
                tokens = [token.lower() for token in sent["tokens"]]
                input_sentences.append(tokens)
            sent_embeddings = sif_embedding.embed_sentences(input_sentences)
            svd = TruncatedSVD(n_components=3, n_iter=50)
            svd.fit(sent_embeddings.numpy())
            example["principal_components"] = svd.components_.tolist()
            for i, sent_embed in enumerate(sent_embeddings):
                example["inputs"][i]["embedding"] = sent_embed.tolist()
            fp.write(json.dumps(example))
            fp.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spensum-data-path", type=str, default=None)
    parser.add_argument(
        "--nyt-train-inputs-path", type=str, default=None)
    parser.add_argument(
        "--nyt-train-abstracts-path", type=str, default=None)
    parser.add_argument(
        "--nyt-test-inputs-path", type=str, default=None)
    parser.add_argument(
        "--nyt-test-abstracts-path", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=43929524)
    parser.add_argument(
        "--nprocs", type=int, default=None)

    args = parser.parse_args()
    if args.nprocs is None:
        args.nprocs = max(1, mp.cpu_count() // 2)

    ntp.set_random_seed(args.seed)

    if args.spensum_data_path is None:
        args.spensum_data_path  = os.getenv("SPENSUM_DATA", None)
        if args.spensum_data_path is None:
            sys.stderr.write(
                "Set SPENSUM_DATA to set location to write data.\n")
            sys.exit(1)

    if args.nyt_train_inputs_path is None:
        args.nyt_train_inputs_path  = os.getenv(
            "NYT_TRAIN_INPUTS_ORIGINAL", None)
        if args.nyt_train_inputs_path is None:
            sys.stderr.write(
                "Set NYT_TRAIN_INPUTS_ORIGINAL to location of " \
                "NYT preprocessed training inputs directory (see " \
                "(https://github.com/gregdurrett/berkeley-doc-summarizer " \
                ").\n")
            sys.exit(1)

    if args.nyt_train_abstracts_path is None:
        args.nyt_train_abstracts_path  = os.getenv(
            "NYT_TRAIN_ABS_ORIGINAL", None)
        if args.nyt_train_abstracts_path is None:
            sys.stderr.write(
                "Set NYT_TRAIN_ABS_ORIGINAL to location of " \
                "NYT preprocessed training abstracts directory (see " \
                "(https://github.com/gregdurrett/berkeley-doc-summarizer " \
                ").\n")
            sys.exit(1)

    if args.nyt_test_inputs_path is None:
        args.nyt_test_inputs_path  = os.getenv(
            "NYT_TEST_INPUTS_ORIGINAL", None)
        if args.nyt_test_inputs_path is None:
            sys.stderr.write(
                "Set NYT_TEST_INPUTS_ORIGINAL to location of " \
                "NYT preprocessed testing inputs directory (see " \
                "(https://github.com/gregdurrett/berkeley-doc-summarizer " \
                ").\n")
            sys.exit(1)

    if args.nyt_test_abstracts_path is None:
        args.nyt_test_abstracts_path  = os.getenv(
            "NYT_TEST_ABS_ORIGINAL", None)
        if args.nyt_test_abstracts_path is None:
            sys.stderr.write(
                "Set NYT_TEST_ABS_ORIGINAL to location of " \
                "NYT preprocessed testing abstracts directory (see " \
                "(https://github.com/gregdurrett/berkeley-doc-summarizer " \
                ").\n")
            sys.exit(1)

    nyt_sds_data_root = os.path.join(args.spensum_data_path, "nyt-sds")

    train_ids, valid_ids = make_train_valid_list(args.nyt_train_inputs_path)
    print("Reading training abstracts...")
    train_abstracts = read_inputs(args.nyt_train_abstracts_path, train_ids)

    train_abstracts = [ex for ex in train_abstracts
                       if sum([s["word_count"] for s in ex["inputs"]]) > 50]
    train_abstracts = train_abstracts[:25000]
    train_ids = [ex["id"] for ex in train_abstracts]
    print(len(train_abstracts))

    print("Writing training reference abstracts...")
    summaries_train_path = os.path.join(        
        nyt_sds_data_root, "summaries", "train", "human_abstracts")
    write_summaries(train_abstracts, summaries_train_path)

    print("Reading validation abstracts...")
    valid_abstracts = read_inputs(args.nyt_train_abstracts_path, valid_ids)
    
    valid_abstracts = [ex for ex in valid_abstracts
                       if sum([s["word_count"] for s in ex["inputs"]]) > 50]
    valid_abstracts = valid_abstracts[:2500]
    valid_ids = [ex["id"] for ex in valid_abstracts]
    print(len(valid_abstracts))

    print("Writing validation reference abstracts...")
    summaries_valid_path = os.path.join(        
        nyt_sds_data_root, "summaries", "valid", "human_abstracts")
    write_summaries(valid_abstracts, summaries_valid_path)

    print("Reading test abstracts...")
    test_abstracts = read_inputs(args.nyt_test_abstracts_path)
    
    test_abstracts = [ex for ex in test_abstracts
                      if sum([s["word_count"] for s in ex["inputs"]]) > 50]
    test_ids = [ex["id"] for ex in test_abstracts]

    print(len(test_abstracts))

    print("Writing test reference abstracts...")
    summaries_test_path = os.path.join(        
        nyt_sds_data_root, "summaries", "test", "human_abstracts")
    write_summaries(test_abstracts, summaries_test_path)

    print("Reading training inputs...")
    train_inputs_data = read_inputs(args.nyt_train_inputs_path, train_ids)
    print("Reading validation inputs...")
    valid_inputs_data = read_inputs(args.nyt_train_inputs_path, valid_ids)
    print("Reading test inputs...")
    test_inputs_data = read_inputs(args.nyt_test_inputs_path, test_ids)

    print("Writing train labels...")
    train_labels_path = os.path.join(        
        nyt_sds_data_root, "labels", "nyt.sds.labels.seq.rouge-1.sw.train.json"
        )
    train_ranks_path = os.path.join(        
        nyt_sds_data_root, "ranks", "nyt.sds.ranks.seq.rouge-1.sw.train.json")
    generate_extracts(
        train_inputs_data, train_abstracts, "sequential", 1,
        train_labels_path, train_ranks_path, args.nprocs)

    print("Writing valid labels...")
    valid_labels_path = os.path.join(        
        nyt_sds_data_root, "labels", "nyt.sds.labels.seq.rouge-1.sw.valid.json")
    valid_ranks_path = os.path.join(        
        nyt_sds_data_root, "ranks", "nyt.sds.ranks.seq.rouge-1.sw.valid.json")
    generate_extracts(
        valid_inputs_data, valid_abstracts, "sequential", 1,
        valid_labels_path, valid_ranks_path, args.nprocs)

    print("Writing test labels...")
    test_labels_path = os.path.join(        
        nyt_sds_data_root, "labels", "nyt.sds.labels.seq.rouge-1.sw.test.json")
    test_ranks_path = os.path.join(        
        nyt_sds_data_root, "ranks", "nyt.sds.ranks.seq.rouge-1.sw.test.json")
    generate_extracts(
        test_inputs_data, test_abstracts, "sequential", 1,
        test_labels_path, test_ranks_path, args.nprocs)

    print("Collecting sentence tokens...")
    all_training_sents = [[token.lower() for token in sent["tokens"]]
                          for ex in train_inputs_data
                          for sent in ex['inputs']]

    print(len(all_training_sents))
    print("Loading sif embedding model...")
    sif_emb = ntp.models.sentence_embedding.SIFEmbedding.from_pretrained()

    print("Fitting principal component...")
    sif_emb.fit_principle_component(all_training_sents)

    sif_path = os.path.join(
        args.spensum_data_path, nyt_sds_data_root, "sif.bin")
    torch.save(sif_emb, sif_path)


    print("Writing training inputs...")
    inputs_train_path = os.path.join(        
        nyt_sds_data_root, "inputs", "nyt.sds.inputs.train.json")
    generate_inputs(train_inputs_data, sif_emb, inputs_train_path)

    print("Writing validation inputs...")
    inputs_valid_path = os.path.join(        
        nyt_sds_data_root, "inputs", "nyt.sds.inputs.valid.json")
    generate_inputs(valid_inputs_data, sif_emb, inputs_valid_path)

    print("Writing test inputs...")
    inputs_test_path = os.path.join(        
        nyt_sds_data_root, "inputs", "nyt.sds.inputs.test.json")
    generate_inputs(test_inputs_data, sif_emb, inputs_test_path)

def create_labels_ranks(args):
    example, abstract, mode, ngram = args
    input_text = [sent["text"] for sent in example["inputs"]]
            
    summaries = ["\n".join(sent["text"] for sent in abstract["inputs"])]
            
    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_text, summaries, mode=mode, ngram=ngram, remove_stopwords=True)
    labels = [min(r, 1) for r in ranks]
    return example["id"], ranks, pairwise_ranks, labels

def generate_extracts(inputs, abstracts, mode, ngram, labels_path, 
                      ranks_path, nprocs):

    create_parent_dir(labels_path)
    create_parent_dir(ranks_path)
    
    args = [(inp, abs, mode, ngram) for inp, abs in zip(inputs, abstracts)]
    pool = mp.Pool(nprocs)
    with open(labels_path, "w") as lbl_fp, open(ranks_path, "w") as rnk_fp:

        for i, result in enumerate(pool.imap(create_labels_ranks, args), 1):
            id, ranks, pairwise_ranks, labels = result

            lbl_fp.write(json.dumps({"id": id, "labels": labels}))
            lbl_fp.write("\n")
            rnk_fp.write(json.dumps({"id": id, "ranks": ranks, 
                                     "pairwise-ranks": pairwise_ranks}))
            rnk_fp.write("\n")
            sys.stdout.write("{}/{}\r".format(i, len(inputs)))
            sys.stdout.flush()
        print("")

def write_summaries(data, output_dir):
    
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for example in data:
        id = "{}.a.spl".format(example["id"]) 
        text = "\n".join([sent["text"] for sent in example["inputs"]])
        path = os.path.join(output_dir, id)
        with open(path, "w") as fp:
            fp.write(text)

if __name__ == "__main__":
    main()
