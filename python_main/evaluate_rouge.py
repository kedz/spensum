import os
import argparse
from collections import defaultdict
import pandas as pd
import rouge_papier


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-summaries", type=str, required=True)
    parser.add_argument(
        "--system-summaries", nargs="+", type=str, required=True)
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
        sys_ids, sys_paths = read_system_summary_manifest(
            sys_dir, id2paths)
        sys_and_sum_paths = [[spth, id2paths[sid]]
                             for sid, spth in zip(sys_ids, sys_paths)]
        config_text = rouge_papier.util.make_simple_config_text(
            sys_and_sum_paths)
        with rouge_papier.util.TempFileManager() as manager:
            config_path = manager.create_temp_file(config_text) 
            df = rouge_papier.compute_rouge(config_path, max_ngram=4, lcs=True)
            data.append(df[-1:])
            systems.append(sys_name)
    df = pd.concat(data, axis=0)
    df.index = systems
    print(df)

if __name__ == "__main__":
    main()
