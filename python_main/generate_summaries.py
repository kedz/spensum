import argparse
import os

import ntp
import spen

import torch


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
    text_field = predictor_data["text_field"]
    feature_field = predictor_data["feature_field"]
    docset_field = ntp.dataio.field_reader.String("docset_id")
    doc_field = ntp.dataio.field_reader.String("doc_id")

    fields = [feature_field, text_field, docset_field, doc_field]
    sequence_field = ntp.dataio.field_reader.Sequence(fields)
    file_reader = ntp.dataio.file_reader.JSONReader([sequence_field])

    (((features,), (text,), (dsids,), (dids,)), example_lengths), = \
            file_reader.read(args.data)

    dataset = ntp.dataio.Dataset(
        (features, example_lengths, "inputs"),
        (dsids, "docset_id"),
        (dids, "doc_id"),
        (text, "text"),
        batch_size=1,
        shuffle=False, 
        gpu=-1)

    for example in dataset.iter_batch():
        doc_id = example.doc_id[0][0]
        docset_id = example.docset_id[0][0]
        probs = model(example.inputs).data[0]
        _, indices = torch.sort(probs, 0, descending=True)

        words = 0
        summary_text = []
        for index in indices:
            text = example.text[0][index]
            summary_text.append(example.text[0][index])
            words += len(text.split())
            if words >= args.summary_word_length:
                break
        summary_path = os.path.join(
            args.output_dir, "{}.{}.spl".format(docset_id, doc_id).lower())

        with open(summary_path, "w") as fp:
            fp.write("\n".join(summary_text)) 
        
if __name__ == "__main__":
    main()
