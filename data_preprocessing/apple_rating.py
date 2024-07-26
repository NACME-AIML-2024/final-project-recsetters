import pandas as pd

# Load CSV data
df_user = pd.read_csv('data_collection/user.csv')
df_apple_music = pd.read_csv('data_collection/apple_items.csv')

# Assign IDs to DataFrames
df_apple_music['id'] = range(1, len(df_apple_music) + 1)
df_user['id'] = range(1, len(df_user) + 1)

# Save the new dataframes with ids
df_items_with_ids = df_apple_music.copy()
df_users_with_ids = df_user.copy()

# Create a dictionary mapping track IDs to track names
id_to_trackname = pd.Series(df_apple_music['trackName'].values, index=df_apple_music['id']).to_dict()

df_lst = []

for i in range(len(df_user)):
    loved_tracks_str = df_user.loc[i, 'loved_tracks']
    username = df_user.loc[i, 'name']
    
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
        if name in df_apple_music['trackName'].values:
            df_lst.append([username, name, date])

# Create DataFrame from the extracted data
df_result = pd.DataFrame(df_lst, columns=['username', 'track_name', 'date_time'])

# Remove rows where track_name is "Unknown"
df_result = df_result[df_result['track_name'] != 'Unknown']

# Drop duplicates
df_result = df_result.drop_duplicates(subset=['username', 'track_name'], keep='last').reset_index(drop=True)

# Convert 'date_time' to Unix time (assuming 'date_time' is already in Unix format as string)
df_result['date_time'] = pd.to_numeric(df_result['date_time'], errors='coerce')

# Filter interactions based on the given criteria
index_count = df_result['track_name'].value_counts().to_dict()

while True:
    song_lst = [track for track, count in index_count.items() if count < 5]
    df_result = df_result[~df_result['track_name'].isin(song_lst)]
    
    user_interaction_count = df_result['username'].value_counts()
    low_interaction_users = user_interaction_count[user_interaction_count < 5].index.tolist()
    
    if not low_interaction_users and not song_lst:
        break
    
    df_result = df_result[~df_result['username'].isin(low_interaction_users)]
    
    updated_track_count = df_result['track_name'].value_counts()
    index_count = updated_track_count.to_dict()

# Reset index if needed
df_result.reset_index(drop=True, inplace=True)
df_result.to_csv('../data_collection/apple_interactions.csv', index=False)