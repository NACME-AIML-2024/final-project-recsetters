''' 
    In this file:
        - User info is fetched using Last Fm API
        - The user's friends' info is used to populate user_info.csv
'''

import requests
import pandas as pd
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define API credentials and parameters
# API_KEY = ''     change this API key to your own
API_KEY = os.getenv('LASTFM_API_KEY')
BASE_URL = 'http://ws.audioscrobbler.com/2.0/'

# Fetch the user info using our API key
def get_user_info(username):
    params = {
        'method': 'user.getinfo',
        'user': username,
        'api_key': API_KEY,
        'format': 'json'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json().get('user', {})
    else:
        return {}

# Fetch the info of the user's 200 friends, every friend is unique
def fetch_friends_info(username, seen_users, limit=200):
    friends_data = []
    page = 1
    while len(friends_data) < limit:
        params = {
            'method': 'user.getFriends',
            'api_key': API_KEY,
            'format': 'json',
            'user': username,
            'limit': 50,
            'page': page
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            friends = data.get('friends', {}).get('user', [])
            for friend in friends:
                if len(friends_data) >= limit:
                    break
                friend_name = friend.get('name')
                if friend_name not in seen_users:
                    friends_data.append({
                        'name': friend_name,
                        'realname': friend.get('realname'),
                        'country': friend.get('country')
                    })
                    seen_users.add(friend_name)
            total_pages = int(data.get('friends', {}).get('@attr', {}).get('totalPages', 1))
            if page >= total_pages:
                break
            page += 1
        else:
            break
    return friends_data


initial_username = 'RJ'  # Initial user to start fetching data
user_data = []
seen_users = set()
num_users = 1000       # Change this value to modify the number of users fetched

    # Fetch initial user's info
initial_user_info = get_user_info(initial_username)
if initial_user_info:
    user_info = {
        'name': initial_user_info.get('name'),
        'realname': initial_user_info.get('realname'),
        'country': initial_user_info.get('country')
    }
    if user_info['name'] not in seen_users: # If that friend's name is unique, append it
        user_data.append(user_info)
        seen_users.add(user_info['name'])

# Fetch friends of the initial user
friends_data = fetch_friends_info(initial_username, seen_users, limit=200)
user_data.extend(friends_data)

# If less than 1000 users, fetch friends of friends
while len(user_data) < num_users and friends_data:
    random_friend = random.choice(friends_data)
    friends_data = fetch_friends_info(random_friend['name'], seen_users, limit=(num_users - len(user_data)))
    user_data.extend(friends_data)

    # Ensure we have exactly 1000 unique users
    final_user_data = user_data[:num_users]

    # Save to CSV --> user_info.csv is the final csv
    df_user = pd.DataFrame(final_user_data)
    df_user.to_csv('data_collection/DATASET/user_info.csv', index=False)
    print(df_user)