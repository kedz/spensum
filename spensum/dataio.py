import ntp


def init_duc_sds_input_reader(embedding_size):
 
    id_field = ntp.dataio.field_reader.String("id")

    text_field = ntp.dataio.field_reader.String("text")
    pc_field = ntp.dataio.field_reader.DenseVector(
        "principal_components")
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
        [id_field, pc_field, inputs_field])

    return inputs_reader
 
def init_duc_sds_label_reader():
 
    id_field = ntp.dataio.field_reader.String("id")
    label_field = ntp.dataio.field_reader.Label(None, vector_type=float,
        vocabulary=["0", "1"])
    label_field.fit_parameters()
    label_sequence_field = ntp.dataio.field_reader.Sequence(
        [label_field], field="labels")
    label_reader = ntp.dataio.file_reader.JSONReader(
        [id_field, label_sequence_field])
    
    return label_reader


def init_duc_sds_pairwise_rank_reader():
 
    id_field = ntp.dataio.field_reader.String("id")
    rank_field = ntp.dataio.field_reader.DenseVector(None, vector_type=int)
    rank_sequence_field = ntp.dataio.field_reader.Sequence(
        [rank_field], field="pairwise-ranks")
    rank_reader = ntp.dataio.file_reader.JSONReader(
        [id_field, rank_sequence_field])
    return rank_reader

def read_ranks(ranks_path, rank_reader, fit=False):
    if fit:
        rank_reader.fit_parameters(ranks_path)

    ((ids,), ((((ranks,),), lengths))) = rank_reader.read(ranks_path)
    return {"ids": ids, "labels": ranks, "lengths": lengths}

def read_inputs(inputs_path, input_reader, fit=False):
    if fit:
        input_reader.fit_parameters(inputs_path)
    ((ids,), (pcs,), ((inputs, lengths))) = input_reader.read(inputs_path)

    ((texts,), (embeddings,), (word_counts,), (positions,)) = inputs
    return {"ids": ids, "texts": texts, "embeddings": embeddings,
            "principal_components": pcs,  
            "word_counts": word_counts, "positions": positions, 
            "lengths": lengths}

def read_labels(labels_path, label_reader, fit=False):
    if fit:
        label_reader.fit_parameters(labels_path)

    ((ids,), ((((labels,),), lengths))) = label_reader.read(labels_path)
    return {"ids": ids, "labels": labels, "lengths": lengths}





def read_input_label_dataset(inputs_path, labels_path, input_reader, 
                             label_reader, batch_size=1, gpu=-1, shuffle=True,
                             fit=False):

    inputs = read_inputs(inputs_path, input_reader, fit=fit)
    labels = read_labels(labels_path, label_reader, fit=fit)

    # Sanity checks
    for i, (id1, id2) in enumerate(zip(inputs["ids"], labels["ids"])):
        assert id1 == id2
        assert inputs["lengths"][i] == labels["lengths"][i]
        assert len(inputs["texts"][i]) == inputs["lengths"][i]
        assert inputs["lengths"][i] == labels["labels"][i].gt(-1).sum()
        assert inputs["lengths"][i] == inputs["embeddings"][i,:,0].ne(-1).sum()

    layout = [
        ["inputs", [
            ["embedding", "embedding"], 
            ["principal_components", "principal_components"],
            ["word_count", "word_count"],
            ["position", "position"],
            ["length", "length"]]
        ], 
        ["targets", "targets"],
        ["metadata", [
            ["id", "id"],
            ["text","text"]]
        ]
    ] 
    dataset = ntp.dataio.Dataset(
        (inputs["ids"], "id"),
        (inputs["texts"], "text"),
        (inputs["embeddings"], inputs["lengths"], "embedding"),
        (inputs["principal_components"], "principal_components"),
        (inputs["word_counts"], inputs["lengths"], "word_count"),
        (inputs["positions"], inputs["lengths"], "position"),
        (labels["labels"], inputs["lengths"], "targets"),
        (inputs["lengths"], "length"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=inputs["lengths"],
        layout=layout,
        shuffle=shuffle)

    return dataset

def read_input_rank_dataset(inputs_path, ranks_path, input_reader, 
                            rank_reader, batch_size=1, gpu=-1, shuffle=True,
                            fit=False):

    inputs = read_inputs(inputs_path, input_reader, fit=fit)
    ranks = read_ranks(ranks_path, rank_reader, fit=fit)

    # Sanity checks
    for i, (id1, id2) in enumerate(zip(inputs["ids"], ranks["ids"])):
        assert id1 == id2
        #assert inputs["lengths"][i] == labels["lengths"][i]
        #assert len(inputs["texts"][i]) == inputs["lengths"][i]
        #assert inputs["lengths"][i] == labels["labels"][i].gt(-1).sum()
        assert inputs["lengths"][i] == inputs["embeddings"][i,:,0].gt(-1).sum()

    layout = [
        ["inputs", [
            ["embedding", "embedding"], 
            ["word_count", "word_count"],
            ["position", "position"],
            ["length", "length"]]
        ], 
        ["targets", "targets"],
        ["metadata", [
            ["id", "id"],
            ["text","text"]]
        ]
    ] 
    dataset = ntp.dataio.Dataset(
        (inputs["ids"], "id"),
        (inputs["texts"], "text"),
        (inputs["embeddings"], inputs["lengths"], "embedding"),
        (inputs["word_counts"], inputs["lengths"], "word_count"),
        (inputs["positions"], inputs["lengths"], "position"),
        (ranks["labels"], ranks["lengths"], "targets"),
        (inputs["lengths"], "length"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=inputs["lengths"],
        layout=layout,
        shuffle=shuffle)

    return dataset

def read_data(path, file_reader, batch_size, gpu=-1, shuffle=True):
    tensor_data, input_sizes = file_reader.read(path)[0]
    docset = tensor_data[0][0]
    doc = tensor_data[1][0]
    text = tensor_data[2][0]
    embedding = tensor_data[3][0]
    word_count = tensor_data[4][0]
    position = tensor_data[5][0]
    label = tensor_data[6][0]
    layout = [
        ["inputs", [
            ["embedding", "embedding"], 
            ["word_count", "word_count"],
            ["position", "position"]]
        ], 
        ["targets", "targets"],
        ["metadata", [
            ["docset", "docset"],
            ["doc", "doc"],
            ["text","text"]]
        ]
    ] 
    dataset = ntp.dataio.Dataset(
        (docset, "docset"),
        (doc, "doc"),
        (text, "text"),
        (embedding, input_sizes, "embedding"),
        (word_count, input_sizes, "word_count"),
        (position, input_sizes, "position"),
        (label, input_sizes, "targets"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=input_sizes,
        layout=layout,
        shuffle=shuffle)

    return dataset


def read_train_and_validation_data(train_path, valid_path, file_reader,
                                   batch_size, gpu):
    file_reader.fit_parameters(train_path)
    train_data = read_data(train_path, file_reader, batch_size, gpu)
    valid_data = read_data(valid_path, file_reader, batch_size, gpu)
    return train_data, valid_data



