import json
import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pickle
from sklearn.metrics import precision_recall_fscore_support
from keras.models import load_model

# nltk.download("punkt")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def preprocess_text(text):
    """Preprocesses text by lowercasing, tokenizing, removing stopwords and stemming."""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word.isalnum() or isfloat(word) and word not in stop_words
    ]
    return " ".join(filtered_tokens)


# Load JSON data
def load_data(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def text2seq(train_text, test_text, tokenizer_name):
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(train_text)
    input_text_index = (
        tokenizer.word_index
    )  # return dictionary of wordss {'the':1, 'earth':2, 'is':3}

    with open(tokenizer_name + ".pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    max_length = max([len(s.split()) for s in train_text])
    print("max length:", max_length)

    train_sequence = tokenizer.texts_to_sequences(train_text)
    test_sequence = tokenizer.texts_to_sequences(test_text)
    return (train_sequence, input_text_index, test_sequence, max_length)


def to_padding(train_df, test_df):
    # Initialize and fit the tokenizer on claim and evidence separately
    x_claims_seq, x_claims_word_index, y_claims_seq, max_claims_length = text2seq(
        train_df["claim"].tolist(), test_df["claim"].tolist(), "tokenizer_claims"
    )
    x_sents_seq, x_sents_word_index, y_sents_seq, max_sents_length = text2seq(
        train_df["evidence"].tolist(),
        test_df["evidence"].tolist(),
        "tokenizer_evidence",
    )

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
    return embedding_matrix, embedding_dim


def main():
    evidence_data = load_data("data/evidence.json")
    dev_claims_data = load_data("data/dev-claims.json")
    # test_claims = load_data('data/test-claims.json')
    evidence_map = load_data("data/curated/preprocessed_evidence_map.json")

    evidence_keys = np.array(list(evidence_data.keys()))

    data_for_dataframe = []
    for claim_id, claim_details in dev_claims_data.items():
        claim_text = preprocess_text(claim_details["claim_text"])
        eids = claim_details["evidences"]
        data_for_dataframe.append(
            {"claim_id": claim_id, "claim": claim_text, "evidence": eids}
        )

    # Create DataFrame
    dev_claims_df = pd.DataFrame(data_for_dataframe)

    with open("tokenizer_claims.pickle", "rb") as handle:
        claims_tokenizer = pickle.load(handle)

    with open("tokenizer_evidence.pickle", "rb") as handle:
        sents_tokenizer = pickle.load(handle)

    max_claims_length = 35
    max_sents_length = 180

    model = load_model("lstm_evidence_retrieval")  # OR hdf5 file

    test_claims = claims_tokenizer.texts_to_sequences(dev_claims_df["claim"])
    test_sents = sents_tokenizer.texts_to_sequences(evidence_map.values())

    test_claims = pad_sequences(test_claims, maxlen=max_claims_length)
    test_sents = pad_sequences(test_sents, maxlen=max_sents_length)
    print("test claims ", test_claims.shape)
    print("test sents ", test_sents.shape)

    threshold = 12527

    max_out_of_bound = 0
    # Assuming test_sents is a numpy array of sequences
    results = []
    for seq in test_sents:
        if np.any(
            seq >= threshold
        ):  # Check if any element in the sequence is greater than or equal to the threshold
            # print("Out-of-bounds sequence:", np.max(seq))
            max_out_of_bound = max(np.max(seq), max_out_of_bound)

    # for i in range(test_claims.shape[0]):
    #     claim_row = test_claims[i]  # Retrieve one row of test_claims

    #     # Replicate this row to match the number of rows in test_sents
    #     replicated_claims = np.tile(claim_row, (test_sents.shape[0], 1))

    #     # Now, create the dictionary to feed into the model
    #     input_dict = {"claims": replicated_claims, "evidences": test_sents}

    #     # Predict using the model
    #     y_pred = model.predict(input_dict, batch_size=1024)
    #     y_pred = np.asarray(y_pred).round()
    #     results.append(y_pred)
    #     print("Y_PREDICT: ", y_pred)
    #     break
    # Assuming y_labels is properly aligned with these predictions
    # Calculate precision, recall, and F1-score
    # scores = precision_recall_fscore_support(y_labels, y_pred, average='binary')
    # print(f"Score of LSTM for claim {i+1}: Precision={scores[0]}, Recall={scores[1]}, F1-Score={scores[2]}")

    predictions_npy_file = "predictions.npy"
    # np.save(predictions_npy_file, np.array(results))

    results = np.load(predictions_npy_file)

    y_pred = results[0].flatten()

    # Use boolean indexing to select only the elements of evidence_data where y_pred is 1
    filtered_evidence = evidence_keys[y_pred == 1]

    np.save("result.npy", filtered_evidence)
    print(filtered_evidence)


main()
