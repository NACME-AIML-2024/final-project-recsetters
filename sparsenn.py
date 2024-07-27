from typing import List
import torch
import torch.nn as nn
import pdb

class MLP(nn.Module):   # multi layer perceptron, takes input features. Each node represents one feature
    def __init__(self, dims: List[int], add_bias=True, act="gelu", apply_layernorm=False, elemwise_affine=False):   # use gelu activation function
        super().__init__()
        self._activation = self._get_activation(act)    # apply activation function
        self._apply_layernorm = apply_layernorm
        self._elemwise_affine = elemwise_affine # boolean to determine the normalization of each layer
        self._add_bias = add_bias   # boolean to add bias to each layer
        self._model = self._create_model(dims)

    def _create_model(self, dims):
        layers = nn.ModuleList()    # to store the layers of the model
        for i in range(1, len(dims)):
            layer = nn.Linear(dims[i-1], dims[i]) if self._add_bias else nn.Linear(dims[i-1], dims[i], bias=False)
            layers.append(layer)

            if i < len(dims) - 1:
                if self._apply_layernorm:
                    layers.append(nn.LayerNorm(dims[i], elementwise_affine=self._elemwise_affine))

                layers.append(self._activation)
        
        return nn.Sequential(*layers)

    def _get_activation(self, act):
        if act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'mish':
            return nn.Mish()
        elif act == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError


    def forward(self, input):
        return self._model(input)

class DotCompressScoringModel(nn.Module):  
    def __init__(self, input_dim: int, hidden_dims: List[int], act='gelu'):
        super(DotCompressScoringModel, self).__init__()
        self.dot_compress_weight = nn.Parameter(torch.empty(2, input_dim // 2))
        nn.init.xavier_normal_(self.dot_compress_weight)
        
        self.dot_compress_bias = nn.Parameter(torch.zeros(input_dim // 2))

        self.dims = [input_dim] + hidden_dims + [1]
        self.output_layer = MLP(self.dims, apply_layernorm=True, elemwise_affine=True)
    
    def forward(self, set_embeddings, item_embeddings):
        all_embeddings = torch.stack([set_embeddings, item_embeddings], dim=1)  # stack the user and item embeddings as a single vector of 1D
        combined_representation = torch.matmul(all_embeddings, torch.matmul(all_embeddings.transpose(1, 2), self.dot_compress_weight) + self.dot_compress_bias).flatten(1)  # computes the dot product of (all_embeddings^T * weight matrix) dot all_embeddings, then flatten
        output = self.output_layer(combined_representation) # pass the output to the MLP
        return output

class TrackSparseNNUserModel(nn.Module):    # embed the user items 
    def __init__(self,
        num_user_ids,
        num_user_countries,
        num_user_names,
        feat_embed_dim=64,
        output_embed_dim=128,   # output embedding
        combine_op='cat',
    ):
        super(TrackSparseNNUserModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_user_ids, embedding_dim=feat_embed_dim)   # embed the user ids
        self.countries_embeddings = nn.Embedding(num_user_countries, embedding_dim=feat_embed_dim)  # embed the countries
        self.user_name_embeddings = nn.Embedding(num_user_names, embedding_dim=feat_embed_dim)  # embed the usernames

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(3*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    
    def forward(self, user_ids, user_countries, user_names):
        id_embeddings = self.id_embeddings(user_ids)
        countries_embeddings = self.countries_embeddings(user_countries)
        user_name_embeddings = self.user_name_embeddings(user_names)

        combined_rep = torch.cat([id_embeddings, countries_embeddings, user_name_embeddings], dim=1) if self.combine_op == 'cat' else \
            id_embeddings + countries_embeddings + user_name_embeddings
        
        return self.act(self.output_mlp(combined_rep))

class TrackSparseNNItemModel(nn.Module):    # embed the item features
    def __init__(self,
        num_track_ids,
        num_track_artists,
        num_track_tags,
        num_track_names,
        feat_embed_dim=96,  #64
        dense_feat_input_dim=384,#326
        output_embed_dim=192,
        combine_op='cat'
    ):
        super(TrackSparseNNItemModel, self).__init__()
        self.track_id_embeddings = nn.Embedding(num_track_ids, feat_embed_dim)  # embed track ids
        self.artists_embeddings = nn.Embedding(num_track_artists, feat_embed_dim)   # embed artists
        self.tags_embeddings = nn.Embedding(num_track_tags, feat_embed_dim) # embed tags
        self.dense_transform = nn.Linear(dense_feat_input_dim, feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(4*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
        
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    def forward(self, track_ids, track_artists, track_tags, track_names):
        track_id_embeddings = self.track_id_embeddings(track_ids)
        artists_embeddings = self.artists_embeddings(track_artists)
        tags_embeddings = self.tags_embeddings(track_tags)
        dense_embeddings = self.act(self.dense_transform(track_names))




        combined_rep = torch.cat([track_id_embeddings, artists_embeddings, tags_embeddings, dense_embeddings], dim=1) if self.combine_op == 'cat' else \
            track_id_embeddings + artists_embeddings + tags_embeddings + dense_embeddings
        
        return self.act(self.output_mlp(combined_rep))


class TrackSparseNN(nn.Module):
    def __init__(self,
        num_user_ids,
        num_user_countries,
        num_user_names,

        num_track_ids,
        num_track_artists,
        num_track_tags,
        num_track_names,
        feat_embed_dim=96,
        dense_feat_embed_dim=384,
        output_embed_dim=192,
        combine_op='cat',
    ):
        super(TrackSparseNN, self).__init__()
        self.user_embedding_model = TrackSparseNNUserModel(num_user_ids, num_user_countries,num_user_names, 
                                                               feat_embed_dim=feat_embed_dim, 
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)

        self.item_embedding_model = TrackSparseNNItemModel(num_track_ids,num_track_artists,num_track_tags,num_track_names,
                                                               feat_embed_dim=feat_embed_dim,
                                                               dense_feat_input_dim=dense_feat_embed_dim,
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)
        
        self.act = nn.GELU()
        self.scoring_model = DotCompressScoringModel(output_embed_dim, [128, 64])

    def forward(self, 
        user_ids = None, user_countries = None, user_names = None,
        track_ids = None, track_artists = None, track_tags = None, track_names = None,
        user_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        user_embeddings = self.user_embedding_model(user_ids, user_countries, user_names) if user_embeddings_precomputed is None \
            else user_embeddings_precomputed
        
        item_embeddings = self.item_embedding_model(track_ids, track_artists, track_tags, track_names) if item_embeddings_precomputed is None \
            else item_embeddings_precomputed
            
        
        return self.act(self.scoring_model(user_embeddings, item_embeddings))
