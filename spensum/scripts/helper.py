import ntp
import torch
import json
import torch.nn.functional as F
from collections import defaultdict
import rouge_papier
import os

def get_inputs_reader(embedding_size=300):
    id_field = ntp.dataio.field_reader.String("id")
    text_field = ntp.dataio.field_reader.String("text")
    feature_field = ntp.dataio.field_reader.DenseVector(
        "embedding", expected_size=embedding_size)
    word_count_field = ntp.dataio.field_reader.DenseVector(
        "word_count", expected_size=1)
    position_field = ntp.dataio.field_reader.DenseVector(
        "sentence_id", expected_size=1, vector_type=int)
    inputs_field = ntp.dataio.field_reader.Sequence(
        [text_field, feature_field, word_count_field, position_field],
        field="inputs")

    inputs_reader = ntp.dataio.file_reader.JSONReader(
        [id_field, inputs_field])
    return inputs_reader

def read_inputs(path, inputs_reader):
    ((ids,), ((inputs, lengths))) = inputs_reader.read(path)
    ((texts,), (embeddings,), (word_counts,), (positions,)) = inputs
    
    data = {"ids": ids, 
            "texts": texts, 
            "embeddings": embeddings,
            "word_counts": word_counts, 
            "positions": positions,
            "lengths": lengths}
    return data


def read_ranks_data(path):
    ids = []
    ranks = []
    lengths = []
    
    with open(path, "r") as fp:
        for line in fp:
            data = json.loads(line)
            ranks_i = [(i, r) for i, r in enumerate(data["ranks"]) if r > 0]
            ranks_i.sort(key=lambda x: x[1])
            ranks_i = [i for i, r in ranks_i]
            ranks.append(ranks_i)
            ids.append(data["id"])
            lengths.append(len(ranks_i))
    
    max_len = max(lengths)
    
    for rank in ranks:
        if len(rank) < max_len:
            rank.extend([-1] * (max_len - len(rank)))
    
    ranks = torch.LongTensor(ranks)
    lengths = torch.LongTensor(lengths)
    
    data = {"ids": ids,
            "ranks": ranks,
            "lengths": lengths}
    
    return data

def create_pointer_targets(batch):
    stop_idx = batch.inputs.length.data[0] 
    batch_size = batch.targets.length.size(0)
    pointers = F.pad(batch.targets.sequence, (0, 1), 'constant', -1)
    for i in range(batch_size):
        pointers[i, batch.targets.length.data[i]] = stop_idx
    return pointers

def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir):

    model.eval()

    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            batch_size = batch.inputs.sequence.size(0)
            predictions = model.greedy_predict(batch.inputs)
            
            for b in range(batch_size):
                id = batch.metadata.id[b]
                preds = [p for p in predictions.data[b].cpu().tolist() if p > -1]
                summary = "\n".join([batch.metadata.text[b][p] for p in preds])                
            
            #for id, summary in zip(batch.metadata.id, batch.metadata):
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(config_path, max_ngram=2, lcs=False, remove_stopwords=True)
        return df[-1:]
