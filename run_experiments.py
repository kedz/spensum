import os
import argparse
from subprocess import check_output
import json
import pandas as pd

DATA_DIR = os.path.join("/", "home", "kedz", "projects2018", "spensum", "data")
INPUT_DIR = os.path.join(DATA_DIR, "duc_sds_data")
MODEL_DIR = os.path.join(DATA_DIR, "duc_sds_models")
PRED_DIR = os.path.join(DATA_DIR, "duc_sds_predictors")
LABEL_DIR = os.path.join(DATA_DIR, "duc_sds_labels")
SUM_DIR = os.path.join(DATA_DIR, "duc_sds_summaries")
RESULTS_DIR = os.path.join(DATA_DIR, "duc_sds_results")
ROUGE_PATH = os.path.join("/", "home", "kedz", "projects2018", "spensum", 
                          "tools", "RELEASE-1.5.5")

tr_tmp = "python python_main/pretrain/train_{script}_module.py " \
            "--train " + os.path.join(INPUT_DIR, "duc.sds.train.json") + " " \
            "--valid " + os.path.join(INPUT_DIR, "duc.sds.valid.json") + " " \
            "--save-module " + os.path.join(MODEL_DIR, "{model}.bin") + " " \
            "--save-predictor " + os.path.join(PRED_DIR, "{pred}.bin")
pr_tmp = "python python_main/predict_labels.py " \
    "--data " + os.path.join(INPUT_DIR, "duc.sds.{part}.json") + " " \
    "--predictor " + os.path.join(PRED_DIR, "{model}.bin") + " " \
    "--output-path " + os.path.join(LABEL_DIR, "{model}.{part}.tsv")

ge_tmp = "python python_main/generate_summaries.py " \
    "--data " + os.path.join(INPUT_DIR, "duc.sds.{part}.json") + " " \
    "--predictor " + os.path.join(PRED_DIR, "{model}.bin") + " " \
    "--output-dir " + os.path.join(SUM_DIR, "{part}", "{model}")

evl_tmp = "python python_main/evaluate_label_prf.py " \
    "--system-labels " + os.path.join(LABEL_DIR, "{model}.{part}.tsv") + " " \
    "--system-names {model} " \
    "--reference-labels " + os.path.join(LABEL_DIR, "gold.{part}.tsv")

evg_tmp = "python python_main/evaluate_rouge.py " \
    "--system-summaries " + os.path.join(SUM_DIR, "{part}", "{model}") \
    + " --system-names {model} " + \
    "--reference-summaries " \
    + os.path.join(SUM_DIR, "{part}", "human_abstract") + " " \
    "--rouge-dir " + ROUGE_PATH


def train_model(model):
    print("training model: {}".format(model))
        
    if model in ["salience", "position", "word_count",
                 "psalience", "coverage"]:
        if model == "psalience":
            script = "positional_salience"
        else:
            script = model
        cmd = tr_tmp.format(
            script=script, model=model, pred=model)
        os.system(cmd)

def predict_label(model):
    print("predicting labels with model: {}".format(model))
    if model in ["salience", "position", "word_count"
                 "psalience", "coverage"]:
        tr_cmd = pr_tmp.format(
            part="train", model=model)
        os.system(tr_cmd)
        va_cmd = pr_tmp.format(
            part="valid", model=model)
        os.system(va_cmd)
        te_cmd = pr_tmp.format(
            part="test", model=model)
        os.system(te_cmd)

def generate_summary(model):
    print("generating summary with model: {}".format(model))
    if model in ["rand3", "lead3", "position", "word_count", "salience", 
                  "psalience", "coverage", "oracle"]:
        tr_cmd = ge_tmp.format(
            part="train", model=model)
        os.system(tr_cmd)
        va_cmd = ge_tmp.format(
            part="valid", model=model)
        os.system(va_cmd)
        te_cmd = ge_tmp.format(
            part="test", model=model)
        os.system(te_cmd)

def eval_labels(model, save_results):
    print("evaluating labels with model: {}".format(model))
    if model in ["salience", "position", "word_count", "lead3", "rand", 
                 "psalience", "coverage"]:
        tr_cmd = evl_tmp.format(
            part="train", model=model)
        print("duc sds train")
        tr_results = check_output(tr_cmd, shell=True).decode("utf8")
        print(tr_results)
        tr_p, tr_r, tr_f = [float(x) for x in 
                            tr_results.split("\n")[1].split()[1:]]

        va_cmd = evl_tmp.format(
            part="valid", model=model)
        print("duc sds valid")
        va_results = check_output(va_cmd, shell=True).decode("utf8")
        print(va_results)
        va_p, va_r, va_f = [float(x) for x in 
                            va_results.split("\n")[1].split()[1:]]
        
        res = {"valid": {"prec.": va_p, "recall": va_r, "f-meas.": va_f},
               "train": {"prec.": tr_p, "recall": tr_r, "f-meas.": tr_f}}

        if save_results:
            save_label_results(model, res)

def eval_summaries(model, save_results):
    print("evaluating summaries with model: {}".format(model))
    if model in ["salience", "position", "word_count", "lead3", "rand3", 
                 "psalience", "coverage", "oracle"]:
        print("duc sds train")
        tr_cmd = evg_tmp.format(part="train", model=model)
        tr_results = check_output(tr_cmd, shell=True).decode("utf8")
        print(tr_results)
        tr_r1, tr_r2, tr_rlcs = [float(x) for x in 
                                 tr_results.split("\n")[1].split()[1:]]

        print("duc sds valid")
        va_cmd = evg_tmp.format(part="valid", model=model)
        va_results = check_output(va_cmd, shell=True).decode("utf8")
        print(va_results)
        va_r1, va_r2, va_rlcs = [float(x) for x in 
                                 va_results.split("\n")[1].split()[1:]]
        
        res = {"valid": {"rouge 1": va_r1, 
                         "rouge 2": va_r2, 
                         "rouge lcs": va_rlcs},
               "train": {"rouge 1": tr_r1, 
                         "rouge 2": tr_r2, 
                         "rouge lcs": tr_rlcs}}

        if save_results:
            save_rouge_results(model, res)

def save_label_results(model, results):

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    model_path = os.path.join(RESULTS_DIR, "{}.labels.json".format(model))
    if not os.path.exists(model_path):
        with open(model_path, "w") as fp:
            fp.write(json.dumps(results))
   
    else:
        with open(model_path, "r") as fp:
            old_results = json.loads(fp.read())
        if old_results["valid"]["f-meas."] < results["valid"]["f-meas."]:
            with open(model_path, "w") as fp:
                fp.write(json.dumps(results))

def save_rouge_results(model, results):

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    model_path = os.path.join(RESULTS_DIR, "{}.rouge.json".format(model))
    if not os.path.exists(model_path):
        with open(model_path, "w") as fp:
            fp.write(json.dumps(results))
   
    else:
        with open(model_path, "r") as fp:
            old_results = json.loads(fp.read())
        if old_results["valid"]["rouge 2"] < results["valid"]["rouge 2"]:
            with open(model_path, "w") as fp:
                fp.write(json.dumps(results))

def get_label_results(model):
    model_path = os.path.join(RESULTS_DIR, "{}.labels.json".format(model))
    if os.path.exists(model_path):
        with open(model_path, "r") as fp:
            return json.loads(fp.read())
    else:
        return None

def get_rouge_results(model):
    model_path = os.path.join(RESULTS_DIR, "{}.rouge.json".format(model))
    if os.path.exists(model_path):
        with open(model_path, "r") as fp:
            return json.loads(fp.read())
    else:
        return None

def print_label_results():
    train_data = []
    valid_data = []
    models = []
    for model in ["rand", "lead3", "position", "word_count", "salience", 
                  "psalience", "coverage"]:
        r = get_label_results(model)
        if r is not None:
            train_data.append(
                (r["train"]["prec."], 
                 r["train"]["recall"], 
                 r["train"]["f-meas."]))
            valid_data.append(
                (r["valid"]["prec."], 
                 r["valid"]["recall"], 
                 r["valid"]["f-meas."]))
            models.append(model)
    print("duc sds train prf")
    print(pd.DataFrame(
        train_data, columns=["prec.", "recall", "f-meas."], index=models))
    print("")
    print("duc sds valid prf")
    print(pd.DataFrame(
        valid_data, columns=["prec.", "recall", "f-meas."], index=models))

def print_rouge_results():
    train_data = []
    valid_data = []
    models = []
    for model in ["rand3", "lead3", "word_count", "coverage", "salience", 
                  "psalience", "position", "oracle"]:
        r = get_rouge_results(model)
        if r is not None:
            train_data.append(
                (r["train"]["rouge 1"], 
                 r["train"]["rouge 2"], 
                 r["train"]["rouge lcs"]))
            valid_data.append(
                (r["valid"]["rouge 1"], 
                 r["valid"]["rouge 2"], 
                 r["valid"]["rouge lcs"]))
            models.append(model)
    print("duc sds train rouge")
    print(pd.DataFrame(
        train_data, columns=["rouge 1", "rouge 2", "rouge lcs"], index=models))
    print("")
    print("duc sds valid rouge")
    print(pd.DataFrame(
        valid_data, columns=["rouge 1", "rouge 2", "rouge lcs"], index=models))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", nargs="+", type=str, 
        choices=["train", "predict", "generate", "eval_labels", 
                 "eval_summaries", "print_label_results", 
                 "print_rouge_results"], 
        required=True)
    parser.add_argument(
        "-m", nargs="+", type=str, default=[], required=False)
    parser.add_argument(
        "--save", action="store_true", default=False)

    args = parser.parse_args()


    for model in args.m:
        for task in args.t:
            if task == "train":
                train_model(model)
            elif task == "predict":
                predict_label(model)
            elif task == "generate":
                generate_summary(model)
            elif task == "eval_labels":
                eval_labels(model, args.save)
            elif task == "eval_summaries":
                eval_summaries(model, args.save)


    if "print_label_results" in args.t:
        print_label_results()
        print("")

    if "print_rouge_results" in args.t:
        print_rouge_results()
        print("")

if __name__ == "__main__":
    main()
