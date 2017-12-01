import ntp


def initialize_sds_reader(embedding_size):
 
    docset_field = ntp.dataio.field_reader.String("docset_id")
    doc_field = ntp.dataio.field_reader.String("doc_id")
    text_field = ntp.dataio.field_reader.String("text")

    feature_field = ntp.dataio.field_reader.DenseVector(
        "embedding", expected_size=embedding_size)
    word_count_field = ntp.dataio.field_reader.DenseVector(
        "word_count", expected_size=1)
    position_field = ntp.dataio.field_reader.DenseVector(
        "sentence_id", expected_size=1, vector_type=int)

    label_field = ntp.dataio.field_reader.Label("label", vector_type=float)

    fields = [
        docset_field,
        doc_field,
        text_field,
        feature_field,
        word_count_field,
        position_field,
        label_field]
    sequence_field = ntp.dataio.field_reader.Sequence(fields)
    file_reader = ntp.dataio.file_reader.JSONReader([sequence_field])

    return file_reader
    
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
