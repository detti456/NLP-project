import pickle

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class KnnSearch:
    def __init__(self, data=None, num_trees=None, emb_dim=None):
        self.num_trees = num_trees
        self.emb_dim = emb_dim

    def get_embeddings_for_data(self, data_ls):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(data_ls)
        return embeddings

    def get_transfer_questions(self, transfer_data):
        transfer_questions = []
        for index, data in enumerate(transfer_data):
            transfer_questions.append(data["input"])
        return transfer_questions

    def get_top_n_neighbours(self, sentence, data_emb, transfer_data, k):
        # sentence: a test instance
        # data_emb: embedded data questions
        # transfer_data: all train data
        sent_emb = self.get_embeddings_for_data(sentence)
        # data_emb = self.get_embeddings_for_data(transfer_questions)
        top_questions = []

        # print("new_emb", sent_emb.shape, data_emb.shape)
        text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
        results_sims = zip(range(len(text_sims)), text_sims)
        sorted_similarities = sorted(results_sims, key=lambda x: x[1], reverse=True)

        for idx, item in sorted_similarities[:k]:
            top_questions.append(transfer_data[idx])

        return top_questions
