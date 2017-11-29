import os
import argparse
from tempfile import NamedTemporaryFile
import pandas as pd
from collections import defaultdict
from subprocess import check_output
import re

def read_reference_summary_manifest(path):
    id2paths = defaultdict(list)
    for filename in os.listdir(path):
        filename_noext = os.path.splitext(filename)[0]
        id, summarizer = os.path.splitext(filename_noext)
        id2paths[id].append(os.path.abspath(os.path.join(path, filename)))
    return id2paths

def read_system_summary_manifest(path, ref_manifest):

    ids = []
    paths = []
    for filename in os.listdir(path):
        id = os.path.splitext(filename)[0]
        ids.append(id)
        paths.append(os.path.abspath(os.path.join(path, filename)))
        assert ref_manifest.get(id, None) is not None
    assert len(ids) == len(ref_manifest)
    return ids, paths

def run_rouge(rouge_dir, manifest_path):
    args = ["perl", os.path.join(rouge_dir, "ROUGE-1.5.5.pl"),
            "-e", os.path.join(rouge_dir, "data"),
            "-n", "2", 
            "-m -a -l 100 -c 95 -r 1000 -f A -p 0.5 -t 0",
            "-z", "SPL", manifest_path]
    output = check_output(" ".join(args), shell=True).decode("utf8")
    mr1 = re.search(r"X ROUGE-1 Average_R: (\d\.\d+) ", output)
    assert mr1 is not None
    rouge_1 = float(mr1.groups()[0])
    mr2 = re.search(r"X ROUGE-2 Average_R: (\d\.\d+) ", output)
    assert mr2 is not None
    rouge_2 = float(mr2.groups()[0])
    mrl = re.search(r"X ROUGE-L Average_R: (\d\.\d+) ", output)
    assert mrl is not None
    rouge_lcs = float(mrl.groups()[0])
    return rouge_1, rouge_2, rouge_lcs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-summaries", type=str, required=True)
    parser.add_argument(
        "--system-summaries", nargs="+", type=str, required=True)
    parser.add_argument("--rouge-dir", type=str, required=True)
    parser.add_argument(
        "--system-names", type=str, nargs="+", required=False, default=None)
    args = parser.parse_args()

    if args.system_names is None:
        args.system_names = [path[-20:] for path in args.system_summaries]

    if len(args.system_names) != len(args.system_summaries):
        raise Exception("--system-names must have the same number of " \
                        "arguments as --system-summaries")


    data = []
    systems = []
    id2paths = read_reference_summary_manifest(args.reference_summaries)
    for sys_dir, sys_name in zip(args.system_summaries, args.system_names):
        with NamedTemporaryFile(mode="w") as fp:
            sys_ids, sys_paths = read_system_summary_manifest(
                sys_dir, id2paths)
            for sys_id, sys_path in zip(sys_ids, sys_paths):
                fp.write(" ".join([sys_path] + id2paths[sys_id]))
                fp.write("\n")
            fp.flush()
            rouge_1, rouge_2, rouge_lcs = run_rouge(
                args.rouge_dir, fp.name)
            data.append([rouge_1, rouge_2, rouge_lcs]) 
            systems.append(sys_name)
    df = pd.DataFrame(
        data, index=systems, columns=["rouge 1", "rouge 2", "rouge lcs"])
    print(df)
    print("")

if __name__ == "__main__":
    main()
