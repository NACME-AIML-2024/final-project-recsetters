import pandas as pd
import requests


API_KEY = '5fae7208288b136b5a2306d078847c4f'
SHARED_SECRET = '00e92aefc3ae823055da97f8865b6e6f'
APPLICATION_NAME = 'RecSetters'
REGISTERED_TO = 'noahtekle98'


df = pd.read_csv("dataset.csv")

specific_columns = ['artists','track_name']
df_spotify_music=df[specific_columns]

df_spotify_music = df_spotify_music[df_spotify_music['track_name'] != 'Music']

df_spotify_music = df_spotify_music.reset_index(drop=True)
df_spotify_music = df_spotify_music.rename(columns={"artists": "artistName", "track_name": "trackName"})
print(df_spotify_music)

song_tags = []

for i in range(len(df_spotify_music)):
    print(i)
    artist_name = df_spotify_music.loc[i, 'artistName']
    track_name = df_spotify_music.loc[i, 'trackName']
    if track_name in df_spotify_music:
        pass
    else:
        api_url_user_info = f'http://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={artist_name}&track={track_name}&api_key={API_KEY}&format=json'
        response_user_info = requests.get(api_url_user_info)

        if response_user_info.status_code == 200:
            data_user_info = response_user_info.json()

            if 'toptags' in data_user_info and 'tag' in data_user_info['toptags']:
                tags = data_user_info['toptags']['tag']
                if tags:
                    first_tag = tags[0]['name']
                    song_tags.append(first_tag)
                else:
                    song_tags.append(None)
            else:
                song_tags.append(None)
        else:
            song_tags.append(None)



df_spotify_music['tags'] = song_tags

df_spotify_music = df_spotify_music[pd.notnull(song_tags)]
df_spotify_music = df_spotify_music.reset_index(drop=True)

df_spotify_music = df_spotify_music.drop_duplicates(subset=['artistName', 'trackName'])
df_spotify_music = df_spotify_music.reset_index(drop=True)
df_spotify_music.to_csv('spotify_songs.csv', index=False)