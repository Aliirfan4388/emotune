import google.generativeai as genai
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotify_config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE , API_KEY_GEMINI
import os

def extract_song_titles(response_text):
    """Extracts song titles from Gemini's response text."""
    lines = response_text.strip().split('\n')
    song_titles = [line.strip().lstrip('* ').lstrip('- ').lstrip('0123456789. ') for line in lines if line.strip()]
    print()
    return song_titles

def configure_gemini_api():
    """Configure the Gemini API with the API key from Streamlit secrets."""
    try:
    
        api_key = API_KEY_GEMINI
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        return False

def fetch_songs(emotion, additional_context=""):
    """
    Fetch song recommendations based on emotion and additional context.
"""
    # Configure Gemini API
    if not configure_gemini_api():
        return []
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Create the prompt based on emotion and context
    prompt = f"Recommend songs that match this emotion: {emotion}"
    if additional_context:
        prompt += f". Additional context: {additional_context}."
    prompt += " Only list the songs without numbering, no introduction, no explanation, one song per line."
    
    # Generate response and extract song titles
    try:
        response = model.generate_content(prompt)
        song_titles = extract_song_titles(response.text)
        return song_titles[:5]  # Return only 5 songs
    except Exception as e:
        st.error(f"Error fetching song recommendations: {e}")
        return []

def add_to_spotify(songs, playlist_name):
    # 1. Check if user is logged in (i.e., if we have a Spotify client in session)
    if "spotify_client" not in st.session_state or st.session_state.spotify_client is None:
        st.error("You must log in first before adding songs to Spotify.")
        return

    # 2. Reuse the existing client
    sp = st.session_state.spotify_client

    # 3. Get user ID
    try:
        current_user = sp.current_user()
        user_id = current_user['id']
        st.write(f"Authenticated as user: {user_id}")
    except Exception as e:
        st.error(f"Error getting current user: {e}")
        return

    # 4. Create playlist
    try:
        new_playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=True,  # or False for private
            description="A playlist created using Python and Spotipy."
        )
        playlist_id = new_playlist['id']
        st.write(f"Created new playlist: {playlist_name} (ID: {playlist_id})")
    except Exception as e:
        st.error(f"Error creating playlist: {e}")
        return

    # 5. Search for and add each song
    for song in songs:
        try:
            results = sp.search(q=song, limit=1)
            if results['tracks']['items']:
                track_id = results['tracks']['items'][0]['id']
                st.write(f"Found track: {song} (ID: {track_id})")
                sp.playlist_add_items(playlist_id, [track_id])
                st.write(f"Added '{song}' to playlist '{playlist_name}'!")
            else:
                st.write(f"Could not find the song: {song}")
        except Exception as e:
            st.error(f"Error adding '{song}': {e}")
