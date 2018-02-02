import argparse
import os
import json
from collections import defaultdict
import rouge_papier


def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument(
        "--remove-stopwords", action="store_true", required=False, 
        default=False)
    parser.add_argument("--reference-summary-dir", type=str, required=True)

    args = parser.parse_args(args)
    ids2refs = collect_reference_paths(args.reference_summary_dir)
    
    with rouge_papier.util.TempFileManager() as manager:
        data_paths = []
        with open(args.inputs, "r") as fp:
            for line in fp:
                example = json.loads(line)

                lines = []
                word_count = 0
                for sent in example["inputs"]:
                    lines.append(sent["text"])
                    word_count += sent["word_count"]
                    if word_count > 100:
                        break
                summary = "\n".join(lines)

                summary_path = manager.create_temp_file(summary)
                data_paths.append([summary_path, ids2refs[example["id"]]])
        config_text = rouge_papier.util.make_simple_config_text(data_paths)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, remove_stopwords=args.remove_stopwords)

    result = df[-1:]
    result.index = ["lead"]
    print(result)
    return result


if __name__ == "__main__":
    main()
