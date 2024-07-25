from collections import defaultdict
from typing import Dict, Hashable, List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pycountry
import ast

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

def read_music(df_users_with_ids, df_items_with_ids):
    df_interactions = pd.read_csv('final_interactions.csv')

    # Process df_users_with_ids
    df_users_with_ids['country'] = df_users_with_ids['country'].apply(lambda x: x.split(',')[-1].strip())
    df_users_with_ids = df_users_with_ids[df_users_with_ids['country'].apply(is_valid_country)]
    
    # Apply the function to the loved_tracks column
    df_users_with_ids['loved_tracks'] = df_users_with_ids['loved_tracks'].apply(extract_track_names)

    usernames = df_users_with_ids['name'].tolist()
    countries = df_users_with_ids['country'].tolist()
    loved_tracks = df_users_with_ids['loved_tracks'].tolist()

    countries_reindexer = convert_to_categorical(countries)
    userids_reindexer = convert_to_categorical(usernames)

    df_items_with_ids['id'] = range(1, len(df_items_with_ids) + 1)
    track_to_id = {row['trackName']: row['id'] for index, row in df_items_with_ids.iterrows()}
    id_to_track = {v: k for k, v in track_to_id.items()}
    track_ids_in_track_to_id = list(track_to_id.values())

    missing_track_ids_in_items = set(track_ids_in_track_to_id) - set(df_items_with_ids['id'].tolist())
    print(f"Missing Track IDs in items.csv: {missing_track_ids_in_items}")

    user_data = {}
    for i in range(len(usernames)):
        reindexed_user_id = userids_reindexer[usernames[i]]
        reindexed_country = countries_reindexer[countries[i]]
        reindexed_loved_tracks = [track_to_id[track] for track in loved_tracks[i] if track in track_to_id]

        user_data[reindexed_user_id] = {
            'user id': reindexed_user_id,
            'country': reindexed_country,
            'loved_tracks': reindexed_loved_tracks
        }

    user_data_df = pd.DataFrame.from_dict(user_data, orient='index')

    track_ids = df_items_with_ids['id'].tolist()
    artist_list = df_items_with_ids['artistName'].tolist()
    tags_list = df_items_with_ids['tags'].tolist()
    track_list = df_items_with_ids['trackName'].tolist()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    title_embeddings = model.encode(track_list)

    track_ids_reindexer = convert_to_categorical(track_ids)
    artist_reindexer = convert_to_categorical(artist_list)
    tags_reindexer = convert_to_categorical(tags_list)

    item_data = {}
    for i in range(len(track_list)):
        reindexed_track_id = track_ids_reindexer[track_ids[i]]
        reindexed_artist = artist_reindexer[artist_list[i]]
        reindexed_tags = tags_reindexer[tags_list[i]]
        reindexed_track_names = title_embeddings[i]
    
        item_data[reindexed_track_id] = {
            'Track id': reindexed_track_id,
            'Track Name': reindexed_track_names,
            'Artist': reindexed_artist,
            'Tags': reindexed_tags
        }

    item_data_df = pd.DataFrame.from_dict(item_data, orient='index')

    reindexed_interactions = defaultdict(list)
    missing_track_ids = []


    userids_reindexer = {value: key for key, value in userids_reindexer.items()}


    for interaction in df_interactions.itertuples():
        try:
            user_id = userids_reindexer[int(interaction.user_id)]
            track_id = track_ids_reindexer[int(interaction.track_id)]
            date_time = int(interaction.date_time)
            reindexed_interactions[user_id].append((track_id, date_time))
        except KeyError as e:
            missing_track_ids.append(interaction.track_id)
            continue

    sorted_reindexed_interactions_by_key = {k: reindexed_interactions[k] for k in sorted(reindexed_interactions)}

    interaction_data = []
    for user_id, track_data in sorted_reindexed_interactions_by_key.items():
        for track_id, date_time in track_data:
            interaction_data.append({'username': user_id, 'track_name': track_id, 'date_time': date_time})

    interaction_data_df = pd.DataFrame(interaction_data)
    interaction_data_df.to_csv('preunknown.csv', index=False)
    interaction_data_df['track_name'] = interaction_data_df['track_name'].apply(lambda x: id_to_track.get(x, 'Unknown'))
    
    # Identify users with 'Unknown' track names
    users_with_unknown = interaction_data_df[interaction_data_df['track_name'] == 'Unknown']['username'].unique()


    # Drop all rows for these users
    interaction_data_df = interaction_data_df[interaction_data_df['track_name'] != 'Unknown']
    interaction_data_df.to_csv('beforefilterpt2.csv', index=False)

    
    # Ensure each user has at least 5 songs and each song has at least 5 listeners
    while True:
        # Filter out songs listened by fewer than 5 users
        track_count = interaction_data_df['track_name'].value_counts()
        tracks_to_remove = track_count[track_count < 5].index
        interaction_data_df = interaction_data_df[~interaction_data_df['track_name'].isin(tracks_to_remove)]

        # Filter out users who have fewer than 5 songs
        user_count = interaction_data_df['username'].value_counts()
        users_to_remove = user_count[user_count < 5].index
        interaction_data_df = interaction_data_df[~interaction_data_df['username'].isin(users_to_remove)]

        # Recompute counts
        track_count = interaction_data_df['track_name'].value_counts()
        user_count = interaction_data_df['username'].value_counts()

        # Check if any more filtering is needed
        if (track_count < 5).sum() == 0 and (user_count < 5).sum() == 0:
            break


    return user_data_df, item_data_df, interaction_data_df, sorted_reindexed_interactions_by_key, df_interactions

df_users_with_ids = pd.read_csv('user.csv')
df_items_with_ids = pd.read_csv('items.csv')
user_data_df, item_data_df, interaction_data_df, sorted_interactions,df_interactions = read_music(df_users_with_ids, df_items_with_ids)
print(interaction_data_df)
interaction_data_df.to_csv('output.csv', index=False)


# this is broken n I dunno why but its unneeded
# def print_interactions(interactions, user_data, item_data):
#     try:
#         for user_id, track_ids in list(interactions.items())[:5]:
#             user_info = user_data.loc[user_id].to_dict()
#             print(f"User {user_info['user id']} (Country: {user_info['country']}) has interacted with tracks:")
#             for track_id, date_time in track_ids:
#                 track_info = item_data.loc[track_id].to_dict()
#                 print(f"  Track ID: {track_id}, Artist: {track_info['Artist']}, Tags: {track_info['Tags']}, Date Time: {date_time}")
#     except Exception as e:
#         print(f"Error: {e}")

# print_interactions(sorted_interactions, user_data_df, item_data_df)