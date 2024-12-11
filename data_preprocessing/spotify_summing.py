import pandas as pd
pt1=pd.read_csv('data_collection/spotify_summary.csv')
pt2=pd.read_csv('data_collection/spotify_items.csv')
pt3=pd.read_csv('data_collection/user.csv')
pt4=pd.read_csv('data_collection/spotify_interactions.csv')

pt2['track_id'] = range(1, len(pt2) + 1)
pt3['user_id'] = range(0, len(pt3))

id_to_track_name = pd.Series(pt2['trackName'].values, index=pt2['track_id']).to_dict()
id_to_user_name = pd.Series(pt3['name'].values, index=pt3['user_id']).to_dict()

pt4['trackName'] = pt4['track_id'].map(id_to_track_name)

# Drop the old track_id column if desired
pt4 = pt4.drop(columns=['track_id'])

pt4['name'] = pt4['user_id'].map(id_to_user_name)

# Drop the old track_id column if desired
pt4 = pt4.drop(columns=['user_id'])

print(pt4)



pt2['song_summary'] = pt1['song_summary']
print(pt2)
new_df=pt2.drop(columns=['trackName'])
print(new_df)
new_df.to_csv('items_with_summary.csv', index=False)

dct={}
for i in range(len(pt2)):
    dct[pt2['trackName'][i]]=new_df['song_summary'][i]


pt4['song_summary'] = pt4['trackName'].map(dct)
print(pt4)

while True:
        # Filter out songs listened by fewer than 5 users
        track_count = pt4['trackName'].value_counts()
        tracks_to_remove = track_count[track_count < 5].index
        pt4 = pt4[~pt4['trackName'].isin(tracks_to_remove)]

        # Filter out users who have fewer than 5 songs
        user_count = pt4['name'].value_counts()
        users_to_remove = user_count[user_count < 5].index
        pt4 = pt4[~pt4['name'].isin(users_to_remove)]

        # Recompute counts
        track_count = pt4['trackName'].value_counts()
        user_count = pt4['name'].value_counts()

        # Check if any more filtering is needed
        if (track_count < 5).sum() == 0 and (user_count < 5).sum() == 0:
            break

print(pt4)
pt4.to_csv('data_collection/spotifies_interactions.csv', index=False)