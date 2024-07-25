import pandas as pd
import requests
import random
import string

# Define API credentials and parameters
API_KEY = '5fae7208288b136b5a2306d078847c4f'

# Fetch the user info using our API key
def get_user_info(username):
    api_url_user_info = f'http://ws.audioscrobbler.com/2.0/?method=user.getinfo&user={username}&api_key={API_KEY}&format=json'
    response = requests.get(api_url_user_info)
    if response.status_code == 200:
        return response.json().get('user', {})
    else:
        return {}

# Fetch the info of the user’s friends, ensuring each friend is unique
def fetch_friends_info(username, seen_users, limit=200):
    friends_data = []
    page = 1
    while len(friends_data) < limit:
        api_url_user_info = f'http://ws.audioscrobbler.com/2.0/?method=user.getFriends&user={username}&api_key={API_KEY}&format=json&limit=50&page={page}'
        response = requests.get(api_url_user_info)
        if response.status_code == 200:
            data = response.json()
            friends = data.get('friends', {}).get('user', [])
            for friend in friends:
                if len(friends_data) >= limit:
                    break
                friend_name = friend.get('name')
                real_name = friend.get('realname')
                country = friend.get('country')
                if None in (friend_name, real_name, country) or country == "None":
                    continue
                elif friend_name not in seen_users:
                    friends_data.append({
                        'name': friend_name,
                        'realname': real_name,
                        'country': country
                    })
                    seen_users.add(friend_name)  # Add user to seen_users set
            total_pages = int(data.get('friends', {}).get('@attr', {}).get('totalPages', 1))
            if page >= total_pages:
                break
            page += 1
        else:
            break
    return friends_data

# Initialize variables
initial_username = 'RJ'  # Initial user to start fetching data
user_data = []
seen_users = set()
num_users = 8000  # Change this value to modify the number of users fetched

# Fetch initial user’s info
initial_user_info = get_user_info(initial_username)
if initial_user_info:
    user_info = {
        'name': initial_user_info.get('name'),
        'realname': initial_user_info.get('realname'),
        'country': initial_user_info.get('country')
    }
    if user_info['name'] not in seen_users:
        user_data.append(user_info)
        seen_users.add(user_info['name'])

# Fetch friends of the initial user
friends_data = fetch_friends_info(initial_username, seen_users, limit=200)
user_data.extend(friends_data)

# If less than 5000 users, fetch friends of friends
while len(user_data) < num_users and friends_data:
    random_friend = random.choice(friends_data)
    friends_data = fetch_friends_info(random_friend['name'], seen_users, limit=(num_users - len(user_data)))
    user_data.extend(friends_data)

# Ensure we have exactly 5000 unique users
final_user_data = user_data[:num_users]

# Save user data to CSV
df_user = pd.DataFrame(final_user_data)
df_user.to_csv('user_info.csv', index=False)
print(f"Found {len(df_user)} unique users.")

# Fetch and process loved tracks for each user
indices_to_drop = []
for i in range(len(df_user)):
    print(i)
    username = df_user.loc[i, 'name']
    api_url_loved_tracks = f'http://ws.audioscrobbler.com/2.0/?method=user.getlovedtracks&user={username}&api_key={API_KEY}&format=json&limit={200}'
    response_loved_tracks = requests.get(api_url_loved_tracks)

    if response_loved_tracks.status_code == 200:
        # Convert the response to JSON format
        data_loved_tracks = response_loved_tracks.json()
        
        # Initialize list to store track info
        track_list = []

        # Extract track information
        tracks = data_loved_tracks['lovedtracks']['track']
        
        # Check if any track name contains '[•=•]', mark for deletion
        if any('[•=•]' in track['name'] for track in tracks):
            indices_to_drop.append(i)
        elif len(tracks) < 5:
            indices_to_drop.append(i)
        else:
            for track in tracks:
                track_info = {
                    'track_name': track['name'],
                    'artist_name': track['artist']['name'],
                    'date_time': track.get('date')
                }
                track_list.append(track_info)
            
            # Convert track_list to a string for storage
            track_list_str = str(track_list)
        
            # Add track info to the DataFrame
            df_user.at[i, 'loved_tracks'] = track_list_str

    else:
        # If error fetching loved tracks, mark for deletion
        print(f"Error fetching loved tracks for user {username}. Deleting row.")
        indices_to_drop.append(i)

# Drop rows marked for deletion
df_user = df_user.drop(indices_to_drop).reset_index(drop=True)

# Save updated user data to CSV
df_user.to_csv('user.csv', index=False)