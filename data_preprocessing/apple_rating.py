import pandas as pd

# Load CSV data
df_user = pd.read_csv('user.csv')
df_apple_music = pd.read_csv('items.csv')

# Assign IDs to DataFrames
df_apple_music['id'] = range(1, len(df_apple_music) + 1)
df_user['id'] = range(1, len(df_user) + 1)

# Create a dictionary mapping track IDs to track names
id_to_trackname = pd.Series(df_apple_music['trackName'].values, index=df_apple_music['id']).to_dict()

id_lst = []
index_count = {}
df_lst = []

for i in range(len(df_user)):
    loved_tracks_str = df_user.loc[i, 'loved_tracks']
    
    start_marker = "'track_name': '"
    end_marker = "',"
    start_marker_2 = "'uts': '"
    end_marker_2 = "'"
    
    track_names = []
    date_times = []
    
    start_pos = 0
    while True:
        start_pos = loved_tracks_str.find(start_marker, start_pos)
        if start_pos == -1:
            break
        start_pos += len(start_marker)
        end_pos = loved_tracks_str.find(end_marker, start_pos)
        if end_pos == -1:
            break
        track_name = loved_tracks_str[start_pos:end_pos]
        track_names.append(track_name)

        # Find date_time
        date_start = loved_tracks_str.find(start_marker_2, end_pos)
        if date_start == -1:
            date_times.append(None)
        else:
            date_start += len(start_marker_2)
            date_end = loved_tracks_str.find(end_marker_2, date_start)
            if date_end == -1:
                date_times.append(None)
            else:
                date_time = loved_tracks_str[date_start:date_end]
                date_times.append(date_time)

        start_pos = end_pos + len(end_marker)

    for name, date in zip(track_names, date_times):
        print(i)
        if name in df_apple_music['trackName'].values:
            index = df_apple_music.index[df_apple_music['trackName'] == name].tolist()[0]
            id_lst = [i, index + 1]  # Add 1 to match track_id
            if index in index_count:
                index_count[index] += 1
            else:
                index_count[index] = 1
            df_lst.append(id_lst + [date])

# Create DataFrame from the extracted data
df_result = pd.DataFrame(df_lst, columns=['user_id', 'track_id', 'date_time'])
# Filter interactions based on the given criteria
df_duplicates = df_result[df_result.duplicated(subset=['user_id', 'track_id'], keep=False)]

df_result = df_result[~df_result.set_index(['user_id', 'track_id']).index.isin(df_duplicates.set_index(['user_id', 'track_id']).index)]

while True:
    song_lst = [key for key, value in index_count.items() if value < 2]
    df_result = df_result[~df_result['track_id'].isin(song_lst)]
    
    user_interaction_count = df_result['user_id'].value_counts()
    low_interaction_users = user_interaction_count[user_interaction_count < 2].index.tolist()
    
    if not low_interaction_users and not song_lst:
        break
    
    df_result = df_result[~df_result['user_id'].isin(low_interaction_users)]
    
    updated_track_count = df_result['track_id'].value_counts()
    index_count = updated_track_count.to_dict()

# Reset index if needed

# Reindex user_id to ensure they are continuous without any gaps
user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(df_result['user_id'].unique()))}
df_result['user_id'] = df_result['user_id'].map(user_id_mapping)

df_result.reset_index(drop=True, inplace=True)
df_result.to_csv('final_interactions.csv', index=False)