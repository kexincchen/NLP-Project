import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors, Word2Vec
nltk.download('punkt')


def preprocess_text(text, remove_stopwords=True):
	"""Preprocesses text by lowercasing, tokenizing, removing stopwords and stemming."""
	stemmer = PorterStemmer()
	tokens = word_tokenize(text.lower())
	if remove_stopwords:
		stop_words = set(stopwords.words('english'))
		filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
	else:
		filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum()]
	return ' '.join(filtered_tokens)
	
def load_data(filepath):
	"""Load JSON data."""
	with open(filepath, 'r') as file:
		data = json.load(file)
	return data

def save_to_json(filepath, data):
	""" Save a dictionary to a JSON file."""
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

def preprocess_evidence(filepath, remove_stopwords=True):
	evidence_data = load_data(filepath)
	evidence_map = {eid: preprocess_text(text, remove_stopwords) for eid, text in evidence_data.items()}
	save_to_json(filepath, evidence_map)


def convert_to_df(data, labelled=True, remove_stopwords=True):
	data_for_dataframe = []
	for claim_id, claim_details in data.items():
		claim_text = preprocess_text(claim_details['claim_text'], remove_stopwords)
		if labelled:
			claim_label = claim_details['claim_label']
			eids = claim_details['evidences']
			data_for_dataframe.append({
					'claim_id': claim_id,
					'claim_text': claim_details['claim_text'],
					'claim_preprocessed': claim_text,
					'evidence': eids,
					'claim_label': claim_label
				})
		else:
			data_for_dataframe.append({
					'claim_id': claim_id,
					'claim_text': claim_details['claim_text'],
					'claim_preprocessed': claim_text,
				})

	# create DataFrame
	df = pd.DataFrame(data_for_dataframe)
	return df

def create_embedding(claims_text, evidence_text, embedding="doc2vec"):
	# convert the claim text into a vector using the specified embedding method
	if embedding == "word2vec":
		w2v = word2vec()
		claim_vec = w2v.get_embedding(claims_text)
		evidence_vec = w2v.get_embedding(evidence_text)
		return claim_vec, evidence_vec
	
	elif embedding == "doc2vec":
		d2v = doc2vec()
		claim_vec = d2v.get_embedding(claims_text)
		evidence_vec = d2v.get_embedding(evidence_text)
		return claim_vec, evidence_vec

	else:
		print("Please choose a embedding type from the following: tfidf, word2vec, doc2vec")

def top_k_evidence(claims_id, claims_emb, evidence_emb, evidence_df, k=3):
	"""
	input:
		claims_id: list of claims' id, (N_c, )
		claims_emb: matrix of claims' embedding, (N_c, d)
		evidence_emb: matrix of evidences' embedding, (N_e, d)
		k: number of evidences selected for each claim

	output:
		top_evidence_id: dictionary that contains claims_id and their corresponding evidences, {claim_id1: [eid1, eid2, ...], claim_id2: []}
	"""
	sim = cosine_similarity(claims_emb, evidence_emb)

	# get top k evidences with highest similarity score with the claim
	data = np.zeros((sim.shape[0], k))
	top_evidence_id = {}
	for i in range(sim.shape[0]):
		data[i] = np.argpartition(sim[i], -k)[-k:]
		top_evidence_id[claims_id[i]] = [evidence_df.iloc[int(ind)]['id'] for ind in data[i]]
	return top_evidence_id
		
class doc2vec:
	def __init__(self):
		self.model = None

	def train_model(self, texts, dimension=300):
		tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(texts)]
		model = Doc2Vec(vector_size=dimension, min_count=1, epochs=20)
		model.build_vocab(tagged_data)
		model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
		model.save("../embedding/d2v.model")
	
	def load_model(self):
		try:
			self.model = Doc2Vec.load("../embedding/d2v.model")
			return self.model
		except:
			print("No model exists. Please train the model first!")

	def get_embedding(self, sent_list, dimension=300):
		# input sentence should be a processed string in which words are separated using splace
		if not self.model:
			self.model = self.load_model()
		
		all_vec = np.zeros((len(sent_list), dimension))
		for i in range(len(sent_list)):
			inferred_vector = self.model.infer_vector(sent_list[i].split())
			all_vec[i] = inferred_vector
		return all_vec

class word2vec:
	def __init__(self):
		self.word_vectors = None

	def train_model(self, texts, dimension=300):
		processed_sentences = [sent.split() for sent in texts]
		model = Word2Vec(
			sentences=processed_sentences,
			vector_size=dimension
		)
		# model.save("word2vec.model")
		word_vectors = model.wv
		word_vectors.save("../embedding/word2vec.wordvectors")

	def load_word_vectors(self):
		try:
			self.word_vectors = KeyedVectors.load('../embedding/word2vec.wordvectors', mmap='r')
			return self.word_vectors
		except:
			print("No model exists. Please train the model first!")

	def get_embedding(self, sent_list, dimension=300):
		if not self.word_vectors:
			self.word_vectors = self.load_word_vectors()

		all_vec = np.zeros((len(sent_list), dimension))
		for i in range(len(sent_list)):
			num_words = len(sent_list[i])
			vec = np.zeros((dimension,))

			# find the word vector for each word in a sentence
			if num_words > 0:
				for word in sent_list[i].split():
					if word in self.word_vectors:
						vec += self.word_vectors[word]
					else:
						num_words -= 1
				# calculate the average word vector
				if num_words > 0:
					vec = np.divide(vec, num_words)
				all_vec[i] = vec
		return all_vec


# read data files
train_claims_data = load_data('../data/train-claims.json')
evidence_data = load_data('../data/evidence.json')
dev_claims_data = load_data('../data/dev-claims.json')
# preprocess_evidence('../data/curated/preprocessed_evidence_map.json', remove_stopwords=True)

# convert data files to dataframe
train_claims_df = convert_to_df(train_claims_data, labelled=True, remove_stopwords=True)
dev_claims_df = convert_to_df(dev_claims_data, labelled=False, remove_stopwords=True)

# evidence_map = load_data('../data/curated/mild_nostopwords_filtered_evidence.json')
evidence_map = load_data('../data/curated/preprocessed_evidence_map.json')
evidence_df = pd.DataFrame(evidence_map.items(), columns=['id', 'evidence'])

train_claims_df['evidence_texts'] = train_claims_df['evidence'].apply(
	lambda x: [evidence_map[evidence_id] for evidence_id in x]
)

train_claims_text = train_claims_df['claim_preprocessed'].tolist()
dev_claims_text = dev_claims_df['claim_preprocessed'].tolist()
dev_claims_id = dev_claims_df['claim_id'].tolist()

evidence_id = list(evidence_map.keys())
evidence_text  = list(evidence_map.values())

# use TFIDF to create embeddings for claims and evidences
vectorizer = TfidfVectorizer()
vectorizer.fit(train_claims_text + evidence_text)
evidence_vec = vectorizer.transform(evidence_text)
dev_claims_vec = vectorizer.transform(dev_claims_text)
print(dev_claims_vec.shape)
print(evidence_vec.shape)


# use word2vec to create embeddings for claims and evidences
# w2v = word2vec()
# # w2v.train_model(train_claims_text + evidence_text)
# dev_claims_vec, evidence_vec = create_embedding(dev_claims_text, evidence_text, embedding='word2vec')

# use doc2vec to create embeddings for claims and evidences
# d2v = doc2vec()
# d2v.train_model(train_claims_text + evidence_text)
# dev_claims_vec, evidence_vec = create_embedding(dev_claims_text, evidence_text, embedding='doc2vec')

# select top 3 evidence for each claim
top_evidence_id = top_k_evidence(dev_claims_id, dev_claims_vec, evidence_vec, evidence_df, k=3)

with open('../data/dev-claims.json', 'r') as input_file:
    test_out_temp = json.load(input_file)

for claim_id, _ in test_out_temp.items():
	test_out_temp[claim_id]["evidences"] = top_evidence_id[claim_id]

with open("dev_predict.json", "w") as outfile:
    json.dump(test_out_temp, outfile)


# Apply on test set
test_claims_data = load_data('../data/test-claims-unlabelled.json')
test_claims_df = convert_to_df(test_claims_data, labelled=False, remove_stopwords=True)
test_claims_text = test_claims_df['claim_preprocessed'].tolist()
test_claims_id = test_claims_df['claim_id'].tolist()

test_claims_vec = vectorizer.transform(test_claims_text)
top_evidence_id = top_k_evidence(test_claims_id, test_claims_vec, evidence_vec, evidence_df, k=3)

test_claims_df['evidences'] = list(top_evidence_id.values())

# get texts of top 5 evidence
test_claims_df['evidence_texts'] = test_claims_df['evidences'].apply(
    lambda x: [evidence_map[evidence_id] for evidence_id in x]
)

# Claim Classification
# combine claim text and evidence texts
X_train = train_claims_df['claim_preprocessed'] + train_claims_df['evidence_texts'].apply(lambda x: ' '.join(x))
y_train = train_claims_df['claim_label']

X_test = test_claims_df['claim_preprocessed'] + test_claims_df['evidence_texts'].apply(lambda x: ' '.join(x))

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_classifier.fit(X_train_count, y_train)
y_pred = rf_classifier.predict(X_test_count)
test_claims_df["claim_label"] = y_pred

test_claims_df.drop(columns=['evidence_texts', 'claim_preprocessed'], inplace=True)
test_claims_df.set_index('claim_id', inplace=True)

# # convert to json file
result = test_claims_df.to_json(orient="index")
with open('test-output.json', 'w') as f:
    f.write(result)