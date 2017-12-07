import os
import argparse

import torch
import spensum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    predictor_data = torch.load(
        args.predictor, map_location=lambda storage, loc: storage)

    module = predictor_data["model"]
    if hasattr(module, "pretrain"):
        module.pretrain()

    file_reader = predictor_data["file_reader"]
    dataset = spensum.dataio.read_data(
        args.data, file_reader, 1, shuffle=False)
    dataset.length_sort = False

    with open(args.output_path, "w") as fp:
        fp.write("id\tlabels\n")
        for example in dataset.iter_batch():
            doc_id = example.metadata.doc[0][0].lower()
            docset_id = example.metadata.docset[0][0].lower()
            if isinstance(module, spensum.model.EnergyModel):
                labels = module.search(example.inputs).data[0].long().tolist()
                label_str = ",".join([str(y) for y in labels])
            else:
                probs = module(example.inputs).data[0]
                label_str = ",".join(str(y) for y in probs.gt(.5).tolist())
            line = "{}.{}\t{}\n".format(docset_id, doc_id, label_str)
            fp.write(line)
        
if __name__ == "__main__":
    main()
