import os
import argparse

import torch
import spensum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--summary-word-length", type=int, default=125)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    predictor_data = torch.load(args.predictor)
    model = predictor_data["model"]
    model.pretrain()

    file_reader = predictor_data["file_reader"]
    dataset = spensum.dataio.read_data(
        args.data, file_reader, 1, shuffle=False)
    dataset.length_sort = False

    for example in dataset.iter_batch():
        doc_id = example.metadata.doc[0][0]
        docset_id = example.metadata.docset[0][0]
        probs = model(example.inputs).data[0]
        _, indices = torch.sort(probs, 0, descending=True)

        words = 0
        summary_text = []
        for index in indices:
            text = example.metadata.text[0][index]
            summary_text.append(text)
            words += len(text.split())
            if words >= args.summary_word_length:
                break
        summary_path = os.path.join(
            args.output_dir, "{}.{}.spl".format(docset_id, doc_id).lower())

        with open(summary_path, "w") as fp:
            fp.write("\n".join(summary_text)) 
        
if __name__ == "__main__":
    main()
