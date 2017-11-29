import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support


def compute_prf(sys_path, ref_path):

    all_sys_labels = []
    all_ref_labels = []
    with open(sys_path, "r") as sys_fp, open(ref_path, "r") as ref_fp:
        sys_fp.readline()
        ref_fp.readline()
        
        for sys_line, ref_line in zip(sys_fp, ref_fp):
            sys_id, sys_label_str = sys_line.strip().split("\t")
            sys_labels = [int(x) for x in sys_label_str.split(",")]
            ref_id, ref_label_str = ref_line.strip().split("\t")
            ref_labels = [int(x) for x in ref_label_str.split(",")]
            
            assert sys_id == ref_id
            assert len(sys_labels) == len(ref_labels)

            all_sys_labels.extend(sys_labels)
            all_ref_labels.extend(ref_labels)
    prec, recall, fmeas, _ = precision_recall_fscore_support(
        all_ref_labels, all_sys_labels, average="binary")
    return prec, recall, fmeas

def main():
    parser = argparse.ArgumentParser(
        "Evaluate precision, recall, and f-measure of extractive sentence " \
        "labeling.")
    parser.add_argument(
        "--system-labels", nargs="+", type=str, required=True)
    parser.add_argument(
        "--reference-labels", type=str, required=True)
    parser.add_argument(
        "--system-names", type=str, nargs="+", required=False, default=None)
    args = parser.parse_args()

    if args.system_names is None:
        args.system_names = [path[-20:] for path in args.system_labels]

    if len(args.system_names) != len(args.system_labels):
        raise Exception("--system-names must have the same number of " \
                        "arguments as --system-labels")

    data = []
    index = []

    for sys_path, sys_name in zip(args.system_labels, args.system_names):
        prec, recall, fmeasure = compute_prf(sys_path, args.reference_labels)
        index.append(sys_name)
        data.append([prec, recall, fmeasure])

    df = pd.DataFrame(
        data, index=index, columns=["precision", "recall", "f-measure"])
    print(df)
    print("")

if __name__ == "__main__":
    main()
