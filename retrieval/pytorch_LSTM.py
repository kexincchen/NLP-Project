import json
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import random
import pickle
from utils import load_data, create_embedding_matrix, preprocess_text, to_padding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


class MyTokenizer:
    def __init__(self):
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_token = {0: "<PAD>", 1: "<UNK>"}

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
                    self.idx_to_token[self.word_index[word]] = word

    def texts_to_sequences(self, text):
        return [
            self.word_index.get(word, self.word_index["<UNK>"]) for word in text.split()
        ]

    def encode(self, text, max_length):
        tokens = self.texts_to_sequences(text)
        if len(tokens) < max_length:
            tokens += [self.word_index["<PAD>"]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def __call__(self, claims, evidences, max_length=512, return_tensors="pt"):
        self.fit_on_texts([claims, evidences])
        encoded_claims = self.encode(claims, max_length)
        encoded_evidences = self.encode(evidences, max_length)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(
                    [encoded_claims, encoded_evidences], dtype=torch.long
                )
            }
        return {"input_ids": [encoded_claims, encoded_evidences]}


# nltk.download('punkt')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

train_claims_data = load_data("data/train-claims.json")
evidence_data = load_data("data/evidence.json")
dev_claims_data = load_data("data/dev-claims.json")
evidence_map = load_data("data/curated/preprocessed_evidence_map.json")
evidence_keys = list(evidence_map.keys())  # List of all evidence IDs

claim_ids = []
for claim_id, claim_details in train_claims_data.items():
    claim_ids.append(claim_id)

# split the claims_df into training and test sets
train, test = train_test_split(claim_ids, test_size=0.2, random_state=42)
len(train)

# train_data_for_dataframe = []
# test_data_for_dataframe = []

# for claim_id, claim_details in train_claims_data.items():
# 	claim_text = preprocess_text(claim_details['claim_text'], stemmer, stop_words)
# 	claim_evidences = set(claim_details['evidences'])  # Convert to set for faster checks

# 	# Add positive examples
# 	for eid in claim_evidences:
# 		evidence_text = evidence_map.get(eid, "NULL")
# 		if evidence_text != "NULL":
# 			data = {
# 				'claim': claim_text,
# 				'evidence': evidence_text,
# 				'label': 1  # Label as relevant
# 			}
# 			if claim_id in train:
# 				train_data_for_dataframe.append(data)
# 			else:
# 				test_data_for_dataframe.append(data)

# 	# Add negative examples
# 	num_neg_samples = min(len(claim_evidences), len(evidence_keys) - len(claim_evidences))  # Limit the number of negative samples
# 	negative_samples = random.sample([k for k in evidence_keys if k not in claim_evidences], num_neg_samples)
# 	for eid in negative_samples:
# 		evidence_text = evidence_map[eid]
# 		data = {
# 			'claim': claim_text,
# 			'evidence': evidence_text,
# 			'label': 0  # Label as not relevant
# 		}
# 		if claim_id in train:
# 				train_data_for_dataframe.append(data)
# 		else:
# 			test_data_for_dataframe.append(data)

# train_df = pd.DataFrame(train_data_for_dataframe)
# test_df = pd.DataFrame(test_data_for_dataframe)

# train_df.to_csv('train_data.csv', index=False)
# test_df.to_csv('test_data.csv', index=False)

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

train_df = train_df.dropna()
test_df = test_df.dropna()

print(train_df.head(10))

tokenizer = MyTokenizer()
(
    x_claim,
    x_sents,
    x_labels,
    x_claims_word_index,
    x_sents_word_index,
    y_claims_data,
    y_sents_data,
    y_labels,
) = to_padding(train_df, test_df, tokenizer)

print("x claim word index ", len(x_claims_word_index))
print("x sent word index ", len(x_sents_word_index))

vocab_size_claims = len(x_claims_word_index) + 2  # +1 for padding, +1 for <UNK>
vocab_size_evidences = len(x_sents_word_index) + 2

