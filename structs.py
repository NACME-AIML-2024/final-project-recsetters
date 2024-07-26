import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.sparse
from tqdm import tqdm
import json
from random import random, shuffle
import random
import pdb

from datareader import df_interactions_processed, df_items_processed, user_data, item_data, sorted_reindexed_interactions_by_key


class InteractionGraph:
    def __init__(self, user_data, item_data, interaction_data) -> None:
        self.user_data = user_data
        self.item_data = item_data
        self.interaction_data = interaction_data
        self.train_edges, self.validation_edges, self.test_edges = [], [], []
        self.adj_matrix: scipy.sparse.dok_matrix = None

    def split_statistics(self):
        training_items = set(self.train_edges[:, 1])
        validation_items = set(self.validation_edges[:, 1])
        test_items = set(self.test_edges[:, 1])

        print("Total number of items = {}".format(len(self.item_data)))
        print("Number of items present across training edges = {}".format(len(training_items)))
        print("Number of items present across val edges = {}".format(len(validation_items)))
        print("Number of items present across test edges = {}".format(len(test_items)))
        print("Average item degree = {}".format(np.mean(self.item_degrees)))
        print("Average user degree = {}".format(np.mean(self.user_degrees)))

        train_val_common_items = training_items.intersection(validation_items)
        train_test_common_items = training_items.intersection(test_items)

        print('Number of items common between train and validation edges = {}'.format(len(train_val_common_items)))
        print('Number of items common between train and test edges = {}'.format(len(train_test_common_items)))

        validation_items = np.array(list(validation_items))
        test_items = np.array(list(test_items))

        num_cold_items_in_val = np.sum(self.is_cold[validation_items])
        num_cold_items_in_test = np.sum(self.is_cold[test_items])

        print('Number of cold items in validation set = {}'.format(num_cold_items_in_val))
        print('Number of cold items in test set = {}'.format(num_cold_items_in_test))


    def create_bipartite_graph(self):
        num_nodes = len(self.user_data) + len(self.item_data) # Num users + num items 
        self.adj_matrix = scipy.sparse.dok_matrix((num_nodes, num_nodes), dtype=bool)  # TODO: Maybe we can optimize with lower precision data types
        
        for edge in self.train_edges:
            self.adj_matrix[edge[0], edge[1]] = 1
            self.adj_matrix[edge[1], edge[0]] = 1

        self.adj_matrix = self.adj_matrix.tocsr()
    
    def compute_tail_distribution(self, warm_threshold):
        self.is_cold = np.zeros((self.adj_matrix.shape[0]), dtype=bool)
        self.start_item_id = len(self.user_data)

        self.user_degrees = np.array(self.adj_matrix[:self.start_item_id].sum(axis=1)).flatten()
        self.item_degrees = np.array(self.adj_matrix[self.start_item_id:].sum(axis=1)).flatten()

        cold_items = np.argsort(self.item_degrees)[:int((1 - warm_threshold) * len(self.item_degrees))] + self.start_item_id
        self.is_cold[cold_items] = True

    def create_data_split(self):
        raise NotImplementedError()

class MusicInteractionGraph(InteractionGraph):
    def __init__(self, user_data, item_data, interaction_data, warm_threshold=0.2) -> None:
        super().__init__(user_data, item_data, interaction_data)
        self.create_data_split()
        self.create_bipartite_graph()
        assert (warm_threshold < 1.0 and warm_threshold > 0.0)
        self.warm_threshold = warm_threshold
        self.compute_tail_distribution()
    
    def create_data_split(self):
            # pdb.set_trace()
            # Leave one out validation - for each user the latest interaction is a test item and the second latest item is the validation item
            #print('Creating data split')
            self.all_edges = set()
            self.train_edges_set = set()
            for user_id in tqdm(self.interaction_data):
                    sorted_interactions = sorted(self.interaction_data[user_id], key=lambda x : x[1])
                    # pdb.set_trace()
                    test_edge = [user_id, sorted_interactions[-1][0]]
                    val_edge = [user_id, sorted_interactions[-2][0]]
                    self.all_edges.add((user_id, sorted_interactions[-2][0]))

                    #print(user_id)

                    train_edges = [[user_id, interaction[0]] for interaction in sorted_interactions[:-2]]
                    for interaction in sorted_interactions[:-2]:
                        self.all_edges.add((user_id, interaction[0]))
                        self.train_edges_set.add((user_id, interaction[0]))           

                    self.train_edges += train_edges
                    self.validation_edges.append(val_edge)
                    self.test_edges.append(test_edge)
            # pdb.set_trace()
            self.train_edges = np.array(self.train_edges)
            self.validation_edges = np.array(self.validation_edges)
            self.test_edges = np.array(self.test_edges)

            # pdb.set_trace()
    
    def compute_tail_distribution(self):
        return super().compute_tail_distribution(self.warm_threshold)

    def __getitem__(self, user_id):
        assert user_id < len(self.user_data), "User ID out of bounds"
        assert isinstance(self.adj_matrix, scipy.sparse.csr_matrix), "Bipartite graph not created: must call create_bipartite_graph first"
        return np.array(self.adj_matrix[user_id, self.start_item_id:].todense()).flatten().nonzero()[0] + self.start_item_id


# Initialize the MusicInteractionGraph with the loaded data
graph = MusicInteractionGraph(user_data, item_data, sorted_reindexed_interactions_by_key, warm_threshold=0.2)

# # Check some of the results
# print("Train edges:\n", graph.train_edges[:5])
# print("Validation edges:\n", graph.validation_edges[:5])
# print("Test edges:\n", graph.test_edges[:5])

# print("Adjacency matrix:\n", graph.adj_matrix[:5])

# # graph.split_statistics()
# import networkx as nx
# import matplotlib.pyplot as plt

# # Sample a smaller subset of interactions for visualization
# sampled_users = random.sample(list(user_data.keys()), 3)  # Sample 3 users
# sampled_items = random.sample(list(item_data.keys()), 3)  # Sample 3 items

# sampled_interaction_data = {user: interactions for user, interactions in sorted_reindexed_interactions_by_key.items() if user in sampled_users}
# sampled_edges = [(user, item) for user in sampled_interaction_data for item, _ in sampled_interaction_data[user] if item in sampled_items]

# # Plotting the interaction graph using NetworkX
# B = nx.Graph()

# # Add nodes with the node attribute "bipartite"
# B.add_nodes_from(sampled_users, bipartite=0)
# B.add_nodes_from(sampled_items, bipartite=1)

# # Add edges (interactions)
# B.add_edges_from(sampled_edges)

# # Plot the bipartite graph
# pos = nx.drawing.layout.bipartite_layout(B, sampled_users)
# plt.figure(figsize=(10, 8))
# nx.draw(B, pos, with_labels=True, node_color=['skyblue' if node in sampled_users else 'lightgreen' for node in B.nodes()], node_size=500, font_size=10)
# plt.title('Sampled Interaction Graph (Bipartite Layout)')
# plt.show()
