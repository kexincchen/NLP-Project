import json
import numpy as np
from nltk.tokenize import word_tokenize
import pickle
import re


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def preprocess_text(text, stemmer=None, stop_words=None):
    # Lowercase the text
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    if stop_words:
        text = ' '.join([word for word in text.split() if word not in stop_words])
    # Apply stemming
    if stemmer:
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Load JSON data
def load_data(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def text2seq(train_text, test_text, tokenizer_name, tokenizer):
    # Fit the tokenizer on the entire dataset

    tokenizer.fit_on_texts(train_text + test_text)

    # Calculate the maximum length of sequences for padding
    max_length = max(
        max([len(text.split()) for text in train_text]),
        max([len(text.split()) for text in test_text]),
    )
    print("Max length:", max_length)

    # Tokenize each text individually
    train_sequence = [
        tokenizer.text_to_sequences(text) for text in train_text
    ]  # Process each text separately
    test_sequence = [
        tokenizer.text_to_sequences(text) for text in test_text
    ]  # Process each text separately
    # Dictionary of words to index
    input_text_index = tokenizer.word_index

    # Optionally, save the tokenizer
    with open(tokenizer_name + ".pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_sequence, input_text_index, test_sequence, max_length


def to_padding(train_df, test_df, tokenizer):
    # Initialize and fit the tokenizer on claim and evidence separately
    x_claims_seq, x_claims_word_index, y_claims_seq, max_claims_length = text2seq(
        train_df["claim"].tolist(),
        test_df["claim"].tolist(),
        "tokenizer_claims",
        tokenizer,
    )
    x_sents_seq, x_sents_word_index, y_sents_seq, max_sents_length = text2seq(
        train_df["evidence_texts"].tolist(),
        test_df["evidence_texts"].tolist(),
        "tokenizer_evidence",
        tokenizer,
    )

    # Ensure all sequences are lists of integers (IDs)
    x_claims_seq = [
        list(map(int, seq))
        for seq in x_claims_seq
        if all(isinstance(x, int) for x in seq)
    ]
    x_sents_seq = [
        list(map(int, seq))
        for seq in x_sents_seq
        if all(isinstance(x, int) for x in seq)
    ]
    x_claims_data = pad_sequences(
        x_claims_seq, maxlen=max_claims_length
    )  # returns array of data
    x_sents_data = pad_sequences(x_sents_seq, maxlen=max_sents_length)
    x_labels = train_df["label"].values

    y_claims_data = pad_sequences(y_claims_seq, maxlen=max_claims_length)
    y_sents_data = pad_sequences(y_sents_seq, maxlen=max_sents_length)
    y_labels = test_df["label"].values

    return (
        x_claims_data,
        x_sents_data,
        x_labels,
        x_claims_word_index,
        x_sents_word_index,
        y_claims_data,
        y_sents_data,
        y_labels,
    )


def create_embedding_matrix(vocab_size, word_vectors, word_index, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                # If a word is not found, the row stays all zeros (or could be initialized randomly)
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
    return embedding_matrix, embedding_dim


def pad_sequences(sequences, maxlen=None):
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    padded_sequences = np.zeros(
        (len(sequences), maxlen), dtype=int
    )  # Ensure dtype is int
    for i, sequence in enumerate(sequences):
        end = min(len(sequence), maxlen)
        padded_sequences[i, :end] = sequence[:end]
    return padded_sequences
