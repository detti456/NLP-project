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


# with open("train_processed.json", "r") as file:
#     train_dataset = json.load(file)
# with open("test_processed.json", "r") as file:
#     test_dataset = json.load(file)
#
# # print(len(test_dataset))
# print("Wrong file")
#
# knn_instance = KnnSearch()
# transfer_questions = knn_instance.get_transfer_questions(train_dataset)
# data_emb = knn_instance.get_embeddings_for_data(transfer_questions)
# few_shot_samples = []
# for index, row in enumerate(test_dataset):
#     exemplars = knn_instance.get_top_n_neighbours(row["input"], data_emb, train_dataset,3)
#     few_shot_samples.extend(exemplars)
#
# print(len(few_shot_samples))
#
# # print(few_shot_samples)
# #
# # print(set(few_shot_samples))
# # print(len(set(few_shot_samples)))
# train_samples = [train_dataset[i] for i in set(few_shot_samples)]
# print(len(train_samples))
#
# with open("train_k_3_samples.json", "w") as file:
#     json.dump(train_samples, file)


# knn_instance = KnnSearch()
# transfer_questions = knn_instance.get_transfer_questions(train_dataset)
# data_emb = knn_instance.get_embeddings_for_data(transfer_questions)
#
#
# with open("train_embeddings", 'wb') as f:
#     pickle.dump(data_emb, f)

# with open("train_embeddings", 'rb') as f:
#     my_list = pickle.load(f)
#
# print(my_list)
#
# print(my_list == data_emb)