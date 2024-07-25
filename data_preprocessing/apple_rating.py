import pandas as pd
import ast


# Load CSV data
df_user = pd.read_csv('../data_collection/DATASET/user.csv')
df_apple_music = pd.read_csv('../data_collection/DATASET/apple_music_dataset.csv')

# Assign IDs to DataFrames
df_apple_music['id'] = range(1, len(df_apple_music) + 1)
df_user['id'] = range(1, len(df_user) + 1)

# Create a dictionary mapping track names to IDs
trackname_to_id = pd.Series(df_apple_music['id'].values, index=df_apple_music['trackName']).to_dict()

id_lst = []
index_count = {}
df_lst = []
skipped_songs = []
missing_users = []

for i in range(len(df_user)):
    loved_tracks_str = df_user.loc[i, 'loved_tracks']
    username = df_user.loc[i, 'name']
    
    start_marker = "'track_name': '"
    end_marker = "',"
    start_marker_2 = "'#text': '"
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
            index = df_apple_music.index[df_apple_music['trackName'] == name].tolist()[0]
            id_lst = [username, name, date]
            if index in index_count:
                index_count[index] += 1
            else:
                index_count[index] = 1
            df_lst.append(id_lst)
        else:
            skipped_songs.append(name)

# Create DataFrame from the extracted data
df_result = pd.DataFrame(df_lst, columns=['username', 'track_name', 'date_time'])

# Remove rows where track_name is "Unknown"
df_result = df_result[df_result['track_name'] != 'Unknown']

df_result = df_result.drop_duplicates(subset=['username','track_name'], keep='last').reset_index(drop=True)

# Filter interactions based on the given criteria
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

# Check if every username in df_result is in df_user
for username in df_result['username'].unique():
    if username not in df_user['name'].values:
        missing_users.append(username)

# Log the missing users
if missing_users:
    print("Missing Users:")
    for user in missing_users:
        print(user)

# Convert 'date_time' to Unix time
df_result['date_time'] = pd.to_datetime(df_result['date_time'], format='%d %b %Y, %H:%M').astype('int64') // 10**9

# Sort the DataFrame by 'username' and 'date_time'
df_result.sort_values(by=['username', 'date_time'], ascending=[True, True], inplace=True)

# Reset index if needed
df_result.reset_index(drop=True, inplace=True)
df_result.to_csv('../data_collection/DATASET/ratings.csv', index=False)

# Print skipped songs
print("Skipped Songs:", len(skipped_songs))

df_result.sort_values(by=['username', 'date_time'], ascending=[True, True], inplace=True)

# Reset index if needed
df_result.reset_index(drop=True, inplace=True)
print('Df result:\n', df_result)