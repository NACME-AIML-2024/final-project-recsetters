from collections import defaultdict
import datetime
from typing import Dict, Hashable, List
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import math
from tqdm import tqdm
import pycountry
import pdb
import torch
import ast

from data_preprocessing.apple_rating import df_users_with_ids, df_items_with_ids

# Helper functions
def convert_to_categorical(input_list, offset=0):
    unique_items = sorted(list(set(input_list)))
    out = {item: i + offset for i, item in enumerate(unique_items)}
    return out

def is_valid_country(country):
    try:
        pycountry.countries.lookup(country)
        return True
    except LookupError:
        return False

def extract_track_names(loved_tracks_str):
    try:
        loved_tracks_list = ast.literal_eval(loved_tracks_str)
        track_names = [track['track_name'] for track in loved_tracks_list]
        return track_names
    except (ValueError, SyntaxError):
        return []

df_interactions = pd.read_csv('data_collection/ratings2.csv')
df_users_with_ids = pd.read_csv('data_collection/user.csv')
df_items_with_ids = pd.read_csv('data_collection/items.csv')
df_items_with_ids['id'] = range(1, len(df_items_with_ids) + 1)
df_users_with_ids['id'] = range(1, len(df_users_with_ids) + 1)


def readmusic(df_user, df_items, df_interactions, user_core=5, item_core=5):
    df_interactions.to_csv('debugging.csv', index=False)
    # Map user names to IDs
    user_name_to_id = {name: user_id for name, user_id in zip(df_user['name'], df_user['id'])}
    df_interactions['user_id'] = df_interactions['username'].map(user_name_to_id)

    # Filter df_user to include only those users present in df_interactions
    valid_usernames = df_interactions['username'].unique()
    df_user = df_user[df_user['name'].isin(valid_usernames)].reset_index(drop=True)

    # Drop duplicates and reset index for items
    df_items = df_items.drop_duplicates(subset=['trackName'], keep='last').reset_index(drop=True)

    # Create a mapping from track names to track IDs
    track_name_to_id = {track: id for track, id in zip(df_items['trackName'], df_items['id'])}

    # Add the track IDs to the interactions dataframe
    df_interactions['track_id'] = df_interactions['track_name'].map(track_name_to_id)

    # Drop interactions with missing track IDs
    df_interactions = df_interactions.dropna(subset=['track_id'])

    # Convert track_id to integer
    df_interactions['track_id'] = df_interactions['track_id'].astype(int)


    # Remove duplicates with the same user id and track id
    df_interactions = df_interactions.drop_duplicates(subset=['user_id', 'track_id'], keep='last').reset_index(drop=True)
    
    # Further filter users and items based on interactions
    valid_users = df_interactions['user_id'].unique()
    valid_items = df_interactions['track_id'].unique()
    df_user = df_user[df_user['id'].isin(valid_users)].reset_index(drop=True)
    df_items = df_items[df_items['id'].isin(valid_items)].reset_index(drop=True)

    #-------- User data processing:
    df_user['country'] = df_user['country'].apply(lambda x: x.split(',')[-1].strip()) # to get country only
    df_user = df_user[df_user['country'].apply(is_valid_country)] # filter out wrong values
    df_user = df_user.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)

    # Ensure valid users are in df_user
    df_user = df_user[df_user['name'].isin(valid_usernames)]

    # Drop interactions whose username is not in df_user
    df_interactions = df_interactions[df_interactions['username'].isin(df_user['name'])]

    # Ensure valid track IDs are in df_items
    df_interactions = df_interactions[df_interactions['track_id'].isin(df_items['id'])]

    # pdb.set_trace()

    # Convert user data to categorical
    username = df_user['name'].to_numpy().astype(str).tolist()
    userid = df_user['id'].to_numpy().astype(str).tolist()
    countries = df_user['country'].to_numpy().astype(str).tolist()

    username_reindexer = convert_to_categorical(username)
    userid_reindexer = convert_to_categorical(userid)
    countries_reindexer = convert_to_categorical(countries)

    user_data = {}
    for i in range(len(username)):
        reindexed_username = username_reindexer[username[i]]
        reindexed_userid = userid_reindexer[userid[i]]
        reindexed_country = countries_reindexer[countries[i]]
        user_data[reindexed_userid] = {
            'username': reindexed_username,
            'country': reindexed_country
        }
        #print(reindexed_username)
    
    #-------- Item data processing:
    track_ids = df_items['id'].to_numpy().astype(str).tolist()
    artist_list = df_items['artistName'].to_numpy().astype(str).tolist()
    tags_list = df_items['tags'].to_numpy().astype(str).tolist()
    track_list = df_items['trackName'].to_numpy().astype(str).tolist()

    track_ids_reindexer = convert_to_categorical(track_ids, offset=len(userid_reindexer))
    tags_reindexer = convert_to_categorical(tags_list)
    artist_reindexer = convert_to_categorical(artist_list)

    # Embed track names using SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    title_embeddings = model.encode(track_list)

    item_data = {}
    for i in range(len(track_list)):
        reindexed_track_id = track_ids_reindexer[track_ids[i]]
        reindexed_tags = tags_reindexer[tags_list[i]]
        reindexed_artist = artist_reindexer[artist_list[i]]
        embedding = title_embeddings[i]
        item_data[reindexed_track_id] = {
            'Track Name': embedding,
            'Artist': reindexed_artist,
            'Tags': reindexed_tags
        }

   #--------- Interaction data processing:
    reindexed_ratings = defaultdict(list)
    for interaction in df_interactions.itertuples():
        try:
            user_id = userid_reindexer[str(interaction.user_id)]
            item_id = track_ids_reindexer[str(interaction.track_id)]
            date_time = interaction.date_time
            reindexed_ratings[user_id].append((item_id, date_time))
        except KeyError as e:
            print(f"KeyError: {e} - User ID: {interaction.user_id}, Track ID: {interaction.track_id}")

    # Sort Reindexed Ratings by key to ensure reproducibility 
    sorted_reindexed_interactions_by_key = {k: reindexed_ratings[k] for k in sorted(reindexed_ratings)}
    
    return df_interactions, df_items, user_data, item_data, sorted_reindexed_interactions_by_key, len(tags_reindexer)

# Run the function to generate and save user_data
df_interactions_processed, df_items_processed, user_data, item_data, sorted_reindexed_interactions_by_key, _ = readmusic(df_users_with_ids, df_items_with_ids, df_interactions)

# Test by checking interactions for user RJ
username = 'RJ'
# user_interactions = df_interactions_processed[df_interactions_processed['username'] == username]
# print(f"Interactions for {username}:")
# print(user_interactions)

# print(df_interactions_processed['username'].value_count())
# look at the unique user data frame id, if that's the first index of the sorted keys its good
# add the unix time

# # Print user data dictionary
# print("User Data Dictionary:")
# print(user_data)

# # Print item data dictionary
# print("Item Data Dictionary:")
# print(item_data)


#------------------Graphs
# import matplotlib.pyplot as plt

# # Plot the distribution of interactions per user
# plt.figure(figsize=(10, 6))
# user_interaction_counts = df_interactions_processed['user_id'].value_counts()
# plt.hist(user_interaction_counts, bins=30, edgecolor='k')
# plt.xlabel('Number of Interactions per User')
# plt.ylabel('Count')
# plt.title('Distribution of Interactions per User')
# plt.show()

# # Plot the distribution of interactions per item
# plt.figure(figsize=(10, 6))
# item_interaction_counts = df_interactions_processed['track_id'].value_counts()
# plt.hist(item_interaction_counts, bins=30, edgecolor='k')
# plt.xlabel('Number of Interactions per Item')
# plt.ylabel('Count')
# plt.title('Distribution of Interactions per Item')
# plt.show()

# import networkx as nx

# # # Create the graph
# # G = nx.Graph()

# # # Add user nodes
# # for user_id in user_data.keys():
# #     G.add_node(user_id, label='user')

# # # Add item nodes
# # for item_id in item_data.keys():
# #     G.add_node(item_id, label='item')

# # # Add edges between users and items
# # for user_id, item_ids in sorted_reindexed_interactions_by_key.items():
# #     for item_id in item_ids:
# #         G.add_edge(user_id, item_id)

# # # Draw the graph
# # plt.figure(figsize=(12, 12))
# # pos = nx.spring_layout(G, k=0.1)
# # nx.draw(G, pos, with_labels=False, node_size=10, node_color='blue', edge_color='gray')
# # plt.title('User-Item Interaction Graph')
# # plt.show()

# import igraph as ig

# g = ig.Graph()

# # Add nodes for users
# for user_id, user_info in user_data.items():
#     g.add_vertex(name=str(user_id), label=user_info['username'], type='user', bipartite=True)

# # Add nodes for items
# for item_id, item_info in item_data.items():
#     g.add_vertex(name=str(item_id), label=str(item_id), type='item', bipartite=False)

# # Add edges for interactions
# edges = []
# for user_id, item_ids in sorted_reindexed_interactions_by_key.items():
#     for item_id in item_ids:
#         edges.append((str(user_id), str(item_id)))

# g.add_edges(edges)

# # Get the bipartite layout
# layout = g.layout_bipartite(types=g.vs["bipartite"])

# color_dict = {"user": "blue", "item": "green"}
# visual_style = {}
# visual_style["vertex_size"] = 20
# visual_style["vertex_label"] = g.vs["label"]
# visual_style["vertex_label_size"] = 10
# visual_style["vertex_color"] = [color_dict[vertex["type"]] for vertex in g.vs]
# visual_style["layout"] = layout
# visual_style["bbox"] = (1200, 800)
# visual_style["margin"] = 100
# visual_style["edge_width"] = 1

# # Save the plot to a file
# ig.plot(g, **visual_style).save('bipartite_graph_plot.png')

# # Display the plot
# ig.plot(g, **visual_style)
