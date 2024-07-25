import pandas as pd
import requests


API_KEY = '5fae7208288b136b5a2306d078847c4f'


df = pd.read_csv("../data_collection/DATASET/apple_music_dataset.csv")
print(df.columns)
specific_columns = ['artistName','trackName']
# Filter rows where trackName is not 'Music'
df_apple_music=df[specific_columns]

df_apple_music = df_apple_music[df_apple_music['trackName'] != 'Music']

df_apple_music = df_apple_music.reset_index(drop=True)
print(df_apple_music)

# Initialize an empty list to store tags
song_tags = []

# Iterate over the first 100 rows (adjust range(len(df_apple_music)) as needed)
for i in range(len(df_apple_music)):
    print(i)
    artist_name = df_apple_music.loc[i, 'artistName']
    track_name = df_apple_music.loc[i, 'trackName']
    if track_name in df_apple_music:
        pass
    else:
        api_url_user_info = f'http://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={artist_name}&track={track_name}&api_key={API_KEY}&format=json'
        response_user_info = requests.get(api_url_user_info)

        if response_user_info.status_code == 200:
        # Convert the response to JSON format
            data_user_info = response_user_info.json()

            # Check if tags are available in the response
            if 'toptags' in data_user_info and 'tag' in data_user_info['toptags']:
                tags = data_user_info['toptags']['tag']
                if tags:
                    # Extract the most popular tag (first tag in the sorted list)
                    first_tag = tags[0]['name']
                    song_tags.append(first_tag)
                else:
                    # Append None if no tags found
                    song_tags.append(None)
            else:
                # Append None if no tags found
                song_tags.append(None)
        else:
            # Append None if API call fails
            song_tags.append(None)



# Add tags to df_apple_music
df_apple_music['tags'] = song_tags

df_apple_music = df_apple_music[pd.notnull(song_tags)]
df_apple_music = df_apple_music.reset_index(drop=True)

df_apple_music = df_apple_music.drop_duplicates(subset=['artistName', 'trackName'])
# Reset index
df_apple_music = df_apple_music.reset_index(drop=True)
# Save to CSV
df_apple_music.to_csv('../data_collection/DATASET/items.csv', index=False)