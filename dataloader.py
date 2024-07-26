import random
import torch
import numpy as np
from torch.utils.data import Dataset
from structs import MusicInteractionGraph
from timeit import default_timer as timer
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

from datareader import readmusic
import pdb
import pandas as pd


# from datareader3 import df_items_processed, user_data, item_data, sorted_reindexed_interactions_by_key
from structs import graph


#region Track Dataset ---------------------------
class TrackDataset(Dataset):
    def __init__(self, igraph : MusicInteractionGraph, mode='train') -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        self.adj_matrix = igraph.adj_matrix
        if mode == 'train':
            self.edges = igraph.train_edges
        elif mode == 'val':
            self.edges = igraph.validation_edges
        else:
            self.edges = igraph.test_edges
    def __len__(self):
        return self.edges.shape[0]
    def __getitem__(self, index):
        return self.edges[index]



class TrackCollator:
    def __init__(self, igraph: MusicInteractionGraph, mode: str, num_neg_samples=1) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        self.adj_matrix = igraph.adj_matrix
        self.num_neg_samples = num_neg_samples
        self.mode = mode
        self.rng = np.random.Generator(np.random.PCG64(seed=0))

    def _generate_in_and_oob_negatives(self, positive_edges):
        item_start_id = len(self.user_data)
        pos_edges = np.array(positive_edges)

        negative_edges = []
        for i, (user_id, _) in enumerate(positive_edges):
            # Out of batch negative Sampling
            candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
            candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

            valid_samples = False
            while not valid_samples:
                neg_items = np.random.choice(candidate_item_probs.shape[0], (self.num_neg_samples,), p=candidate_item_probs)
                for neg_item in neg_items:
                    if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                        valid_samples = False
                        break
                    valid_samples = True
            # In batch negative sampling
            in_batch_candidates = np.concatenate((pos_edges[:i], pos_edges[(i+1):]))[:, 1]
            idxs_to_delete = [idx for idx, candidate in enumerate(in_batch_candidates) if (user_id, candidate) in self.igraph.all_edges]
            valid_candidates = np.delete(in_batch_candidates, idxs_to_delete)
            in_batch_negs = self.rng.choice(valid_candidates, (self.num_neg_samples,), replace=False)
            for neg_item in neg_items:
                negative_edges.append([user_id, neg_item + item_start_id])
            for neg_item in in_batch_negs:
                negative_edges.append([user_id, neg_item])
        return negative_edges
    
    def _get_edges_to_score(self, edges):
        item_start_id = len(self.user_data)
        offsets = []
        edges_to_score = []
        for user_id, _ in edges:
            current_offset = len(edges_to_score)
            offsets.append(current_offset)
            negatives = np.argwhere(np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0).flatten() # The true validation and test items will be a part of this array
            edges_to_score += [[user_id, negative + item_start_id] for negative in negatives]
        return np.asarray(edges_to_score), np.asarray(offsets)
    
    def _fetch_data(self, edges):
        # create id system
        user_feats = [(user_id, self.user_data[user_id]['country'], self.user_data[user_id]['username']) for user_id, _ in edges]
        item_feats = [(track_id, self.item_data[track_id]['Artist'], self.item_data[track_id]['Tags']) for _, track_id in edges]
        item_title_embeddings = [self.item_data[track_id]['Track Name'] for _, track_id in edges]
        # pdb.set_trace()
        return user_feats, item_feats, item_title_embeddings
    
    def __call__(self, positive_edges):
        if self.mode != 'train':
            true_edges = np.array(positive_edges)
            edges_to_score, offsets = self._get_edges_to_score(true_edges)
            return torch.as_tensor(true_edges, dtype=torch.int32), torch.as_tensor(edges_to_score, dtype=torch.int32), torch.as_tensor(offsets, dtype=torch.int32)
        
        negative_edges = self._generate_in_and_oob_negatives(positive_edges)
        edges = positive_edges + negative_edges

        user_feats, item_feats, item_title_embeddings = self._fetch_data(edges)

        user_feats_by_type = list(zip(*user_feats))
        user_ids = torch.as_tensor(user_feats_by_type[0], dtype=torch.int64)
        user_country = torch.as_tensor(user_feats_by_type[1], dtype=torch.int64)
        user_name = torch.as_tensor(user_feats_by_type[2], dtype=torch.int64)

        item_feats_by_type = list(zip(*item_feats))
        track_ids = torch.as_tensor(item_feats_by_type[0], dtype=torch.int64) - len(self.user_data)
        track_artists = torch.as_tensor(item_feats_by_type[1], dtype=torch.int64)
        tag_embeddings = torch.as_tensor(item_feats_by_type[2], dtype=torch.int64)
        track_name_embeddings = torch.as_tensor(np.asarray(item_title_embeddings), dtype=torch.float32)


        return (user_ids, user_country,user_name), (track_ids, track_artists, tag_embeddings, track_name_embeddings)
    
class TrackInferenceItemsDataset(Dataset):
    def __init__(self, igraph : MusicInteractionGraph) -> None:
        self.all_item_ids = sorted(list(igraph.item_data.keys()))
        self.item_reindexer = {}
        for item_id in self.all_item_ids:
            self.item_reindexer[item_id] = len(self.item_reindexer)
        self.reverse_item_indexer = {v : k for k, v in self.item_reindexer.items()}
    def __len__(self):
        return len(self.all_item_ids)
    def __getitem__(self, index):
        return self.all_item_ids[index]
    
class TrackItemsCollator:
    def __init__(self, igraph : MusicInteractionGraph) -> None:
        self.igraph = igraph
        self.item_data = igraph.item_data

    def __call__(self, batch):
        item_feats = [(track_id, self.item_data[track_id]['Artist'], self.item_data[track_id]['Tags']) for track_id in batch]
        item_title_embeddings = [self.item_data[track_id]['Track Name'] for track_id in batch]
        item_feats_by_type = list(zip(*item_feats))
        track_ids = torch.as_tensor(item_feats_by_type[0], dtype=torch.int64)
        zero_indexed_track_ids = track_ids - len(self.igraph.user_data)
        track_artists = torch.as_tensor(item_feats_by_type[1], dtype=torch.int64)
        tag_embeddings = torch.as_tensor(item_feats_by_type[2],  dtype=torch.int64)
        track_name_embeddings = torch.as_tensor(np.asarray(item_title_embeddings), dtype=torch.float32)
        return (track_ids, track_artists, tag_embeddings, track_name_embeddings, zero_indexed_track_ids)
    
class TrackInferenceUsersDataset(Dataset):
    def __init__(self, igraph : MusicInteractionGraph) -> None:
        self.all_user_ids = sorted(list(igraph.user_data.keys()))

    def __len__(self):
        return len(self.all_user_ids)
    
    def __getitem__(self, index):
        return self.all_user_ids[index]
    
class TrackUsersCollator:
    def __init__(self, igraph : MusicInteractionGraph) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data

    def __call__(self, batch):
        user_feats = [(user_id, self.user_data[user_id]['country'],self.user_data[user_id]['username']) for user_id in batch]
        user_feats_by_type = list(zip(*user_feats))
        user_ids = torch.as_tensor(user_feats_by_type[0], dtype=torch.int64)
        user_countries = torch.as_tensor(user_feats_by_type[1], dtype=torch.int64)
        user_names = torch.as_tensor(user_feats_by_type[2], dtype=torch.int64)
        return (user_ids, user_countries,user_names)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
