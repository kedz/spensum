import argparse
import os
import sys
import json
import shutil

def ranks2summary(input, pred):

    indices_ranks = [(i, rank) for i, rank in enumerate(pred["ranks"])
                     if rank > 0]
    indices_ranks.sort(key=lambda x: x[1])
    
    lines = [input["inputs"][index]["text"] for index, rank in indices_ranks]
    text = "\n".join(lines)

    return text 

def labels2summary(input, pred):

    lines = [input["inputs"][index]["text"] 
             for index, label in enumerate(pred["labels"]) if label == 1]
    text = "\n".join(lines)

    return text 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs", type=str, required=True)
    parser.add_argument(
        "-p", "--predictions", type=str, required=True)
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True)
    parser.add_argument(
        "-m", "--mode", type=str, choices=["labels", "ranks"])

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    if args.output_dir != "" and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.inputs, "r") as i_fp, open(args.predictions, "r") as p_fp:
        for l, (input_line, pred_line) in enumerate(zip(i_fp, p_fp)):

            input = json.loads(input_line)
            pred = json.loads(pred_line)

            if input["id"] != pred["id"]:
                sys.stderr.write(
                    "In line {}: {} id {} != {} id {}\n".format(
                        l, args.inputs, input["id"], 
                        args.predictions, pred["id"]))
                sys.stderr.flush()
                sys.exit(1)

            if args.mode == "labels":
                text = labels2summary(input, pred)
            else:
                text = ranks2summary(input, pred)

            path = os.path.join(args.output_dir, "{}.spl".format(input["id"]))
            with open(path, "w") as fp:
                fp.write(text)

if __name__ == "__main__":
    main()
