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
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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

    with open(args.output_path, "w") as fp:
        fp.write("id\tlabels\n")
        for example in dataset.iter_batch():
            doc_id = example.doc_id[0][0].lower()
            docset_id = example.docset_id[0][0].lower()
            probs = model(example.inputs).data[0]
            label_str = ",".join(str(y) for y in probs.gt(.5).tolist())
            line = "{}.{}\t{}\n".format(docset_id, doc_id, label_str)
            fp.write(line)
        
if __name__ == "__main__":
    main()
