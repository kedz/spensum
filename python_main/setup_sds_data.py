import os
import argparse
import json
import random
import numpy as np
import ntp
import tarfile
import math

DUC2001_TRAIN_FILENAME = "duc2001.sds.train.json"
DUC2001_TEST_FILENAME = "duc2001.sds.test.json"
DUC2002_FILENAME = "duc2002.sds.json"
DUC2001_ORIG_TAR = "duc2001-sds-data.tar.gz"
DUC2002_ORIG_TAR = "duc2002-sds-data.tar.gz"

def write_json(data, path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, "w") as fp:
        for example in data:
            for sent in example:
                if math.isnan(sent["embedding"][0]):
                    sent["embedding"] = [0 for x in sent["embedding"]]
            fp.write(json.dumps(example))
            fp.write("\n")
 
def read_duc2001_summaries(tar_path):
    all_summaries = {}
    with tarfile.open(tar_path) as tar_fp:
        for entry in tar_fp:  
            if entry.name.endswith("target.json"):
                summaries = json.loads(
                    tar_fp.extractfile(entry).read().decode("utf8"))
                part = "train" if "train" in entry.name else "test"
                id = "{}.{}".format(
                    summaries[0]["docset_id"].lower(), 
                    summaries[0]["input_ids"][0].lower())
                all_summaries[(id, part)] = summaries
    return all_summaries

def read_duc2002_summaries(tar_path):
    all_summaries = []
    with tarfile.open(tar_path) as tar_fp:
        for entry in tar_fp:  
            if entry.name.endswith("target.json"):
                summaries = json.loads(
                    tar_fp.extractfile(entry).read().decode("utf8"))
                all_summaries.append(summaries)
    return all_summaries

def write_gold_summaries(data, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
   
    for summaries in data:
        for summary in summaries:
            fn = "{}.{}.{}.spl".format(
                summary["docset_id"].lower(), 
                summary["input_ids"][0].lower(),
                summary["summarizer"])
            path = os.path.join(root_path, fn)
            with open(path, "w") as fp:
                fp.write("\n".join(summary["sentences"]))

def write_gold_labels(data, path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
   
    with open(path, "w") as fp:
        fp.write("id\tlabels\n")
        for example in data:
            labels = ",".join([str(sent["label"]) for sent in example])
            id = "{}.{}".format(
                example[0]["docset_id"].lower(), 
                example[0]["doc_id"].lower())
            line = "{}\t{}\n".format(id, labels)
            fp.write(line)

def write_random3_summaries(data, path):
    if not os.path.exists(path):
        os.makedirs(path)
   
    for example in data:
        id = "{}.{}".format(
            example[0]["docset_id"].lower(), 
            example[0]["doc_id"].lower())
        indices = [i for i in range(len(example))]
        random.shuffle(indices)

        with open(os.path.join(path, "{}.spl".format(id)), "w") as fp:
            text = "\n".join([example[i]["text"] for i in indices[:3]])
            fp.write(text)

def write_random_labels(data, path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
   
    with open(path, "w") as fp:
        fp.write("id\tlabels\n")
        for example in data:
            label_vector = np.random.choice([0, 1], size=(len(example),))
            labels = ",".join([str(y) for y in label_vector])
            id = "{}.{}".format(
                example[0]["docset_id"].lower(), 
                example[0]["doc_id"].lower())
            line = "{}\t{}\n".format(id, labels)
            fp.write(line)

def write_lead3_summaries(data, path):
    if not os.path.exists(path):
        os.makedirs(path)
   
    for example in data:
        id = "{}.{}".format(
            example[0]["docset_id"].lower(), 
            example[0]["doc_id"].lower())
        with open(os.path.join(path, "{}.spl".format(id)), "w") as fp:
            text = "\n".join([sent["text"] for sent in example[:3]])
            fp.write(text)

def write_lead3_labels(data, path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
   
    with open(path, "w") as fp:
        fp.write("id\tlabels\n")
        for example in data:

            label_vector = [1] * 3 + [0] * (len(example) - 3)
            labels = ",".join([str(y) for y in label_vector])
            id = "{}.{}".format(
                example[0]["docset_id"].lower(), 
                example[0]["doc_id"].lower())
            line = "{}\t{}\n".format(id, labels)
            fp.write(line)

def write_oracle_summaries(data, path):
    if not os.path.exists(path):
        os.makedirs(path)
   
    for example in data:
        id = "{}.{}".format(
            example[0]["docset_id"].lower(), 
            example[0]["doc_id"].lower())
        summary_texts = []
        for sent in example:
            if sent["label"] == 1:
                summary_texts.append(sent["text"])

        with open(os.path.join(path, "{}.spl".format(id)), "w") as fp:
            fp.write("\n".join(summary_texts))

  

def main():
    parser = argparse.ArgumentParser(
        "Setup single document summarizaton for duc 2001/2002 data.\n" \
        "       Creates a train/validation split for duc 2001 and uses " \
        "duc 2002\n       for test data.")

    parser.add_argument("--raw-data-dir", type=str, required=True)
    parser.add_argument("--output-data-dir", type=str, required=True)
    parser.add_argument("--summary-dir", type=str, required=True)
    parser.add_argument("--extract-label-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=8329221)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    duc2001_train_path = os.path.join(
        args.raw_data_dir, DUC2001_TRAIN_FILENAME)
    duc2001_test_path = os.path.join(
        args.raw_data_dir, DUC2001_TEST_FILENAME)
    duc2001_tar_path = os.path.join(
        args.raw_data_dir, DUC2001_ORIG_TAR)
    duc2002_path = os.path.join(
        args.raw_data_dir, DUC2002_FILENAME)
    duc2002_tar_path = os.path.join(
        args.raw_data_dir, DUC2002_ORIG_TAR)

    if not os.path.exists(duc2001_train_path):
        raise Exception("Could not find file {} in directory {}".format(
            DUC2001_TRAIN_FILENAME, args.raw_data_dir))
    if not os.path.exists(duc2001_test_path):
        raise Exception("Could not find file {} in directory {}".format(
            DUC2001_TEST_FILENAME, args.raw_data_dir))
    if not os.path.exists(duc2001_tar_path):
        raise Exception("Could not find file {} in directory {}".format(
            DUC2001_ORIG_TAR, args.raw_data_dir))
    if not os.path.exists(duc2002_path):
        raise Exception("Could not find file {} in directory {}".format(
            DUC2002_FILENAME, args.raw_data_dir))
    if not os.path.exists(duc2002_tar_path):
        raise Exception("Could not find file {} in directory {}".format(
            DUC2002_ORIG_TAR, args.raw_data_dir))

    duc2001_summaries = read_duc2001_summaries(duc2001_tar_path)
    duc2002_summaries = read_duc2002_summaries(duc2002_tar_path)


    train_dev_data = []
    orig_split = []
    test_data = []

    train_data = []
    train_summaries = []
    valid_data = []
    valid_summaries = []

    with open(duc2001_train_path, "r") as fp:
        for line in fp:
            train_dev_data.append(json.loads(line))
            orig_split.append("train")

    with open(duc2001_test_path, "r") as fp:
        for line in fp:
            train_dev_data.append(json.loads(line))
            orig_split.append("test")
    
    with open(duc2002_path, "r") as fp:
        for line in fp:
            test_data.append(json.loads(line))
    
    train_indices, valid_indices = ntp.trainer.generate_splits(
        [i for i in range(len(train_dev_data))], train_per=.9, valid_per=0)

    for index in train_indices:
        example = train_dev_data[index]
        id = "{}.{}".format(
            example[0]["docset_id"].lower(), example[0]["doc_id"].lower())
        part = orig_split[index]
        summaries = duc2001_summaries[(id, part)]
        train_summaries.append(summaries)
        train_data.append(train_dev_data[index])

    for index in valid_indices:
        example = train_dev_data[index]
        id = "{}.{}".format(
            example[0]["docset_id"].lower(), example[0]["doc_id"].lower())
        part = orig_split[index]
        summaries = duc2001_summaries[(id, part)]
        valid_summaries.append(summaries)
        valid_data.append(train_dev_data[index])

    write_json(
        train_data, os.path.join(args.output_data_dir, "duc.sds.train.json"))
    write_json(
        valid_data, os.path.join(args.output_data_dir, "duc.sds.valid.json"))
    write_json(
        test_data, os.path.join(args.output_data_dir, "duc.sds.test.json"))


    
    write_gold_labels(
        train_data, os.path.join(args.extract_label_dir, "gold.train.tsv"))
    write_random_labels(
        train_data, os.path.join(args.extract_label_dir, "rand.train.tsv"))
    write_lead3_labels(
        train_data, os.path.join(args.extract_label_dir, "lead3.train.tsv"))

    write_gold_labels(
        valid_data, os.path.join(args.extract_label_dir, "gold.valid.tsv"))
    write_random_labels(
        valid_data, os.path.join(args.extract_label_dir, "rand.valid.tsv"))
    write_lead3_labels(
        valid_data, os.path.join(args.extract_label_dir, "lead3.valid.tsv"))

    write_gold_labels(
        test_data, os.path.join(args.extract_label_dir, "gold.test.tsv"))
    write_random_labels(
        test_data, os.path.join(args.extract_label_dir, "rand.test.tsv"))
    write_lead3_labels(
        test_data, os.path.join(args.extract_label_dir, "lead3.test.tsv"))

    write_random3_summaries(
        train_data, os.path.join(args.summary_dir, "train", "rand3"))
    write_lead3_summaries(
        train_data, os.path.join(args.summary_dir, "train", "lead3"))
    write_gold_summaries(
        train_summaries, 
        os.path.join(args.summary_dir, "train", "human_abstract"))
    write_oracle_summaries(
        train_data, 
        os.path.join(args.summary_dir, "train", "oracle"))

    write_random3_summaries(
        valid_data, os.path.join(args.summary_dir, "valid", "rand3"))
    write_lead3_summaries(
        valid_data, os.path.join(args.summary_dir, "valid", "lead3"))
    write_gold_summaries(
        valid_summaries, 
        os.path.join(args.summary_dir, "valid", "human_abstract"))
    write_oracle_summaries(
        valid_data, 
        os.path.join(args.summary_dir, "valid", "oracle"))

    write_random3_summaries(
        test_data, os.path.join(args.summary_dir, "test", "rand3"))
    write_lead3_summaries(
        test_data, os.path.join(args.summary_dir, "test", "lead3"))
    write_gold_summaries(
        duc2002_summaries, 
        os.path.join(args.summary_dir, "test", "human_abstract"))
    write_oracle_summaries(
        test_data, 
        os.path.join(args.summary_dir, "test", "oracle"))




if __name__ == "__main__":
    main()
