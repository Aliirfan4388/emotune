import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotify_config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE

CACHE_PATH = ".cache-spotify"

def authenticate_user():
    st.subheader("Login with your Spotify Account")

    # Create SpotifyOAuth instance
    sp_oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        show_dialog=True,  # forces Spotify to show the login dialog every time
        cache_path=CACHE_PATH
    )

    # Get query parameters using experimental API
    query_params = st.experimental_get_query_params()
    code = query_params.get("code", [None])[0]

    # Check if already authenticated in session state
    if st.session_state.get("spotify_token"):
        st.success(f"Already logged in as {st.session_state['user_info']['display_name']}.")
        return True

    if code:
        try:
            # Exchange code for access token
            token_info = sp_oauth.get_access_token(code, as_dict=True)
            spotify_client = spotipy.Spotify(auth=token_info["access_token"])
            user_info = spotify_client.current_user()

            # Store token, client, and user info in session state
            st.session_state.spotify_token = token_info
            st.session_state.spotify_client = spotify_client
            st.session_state.user_info = user_info

            st.success(f"Logged in as {user_info['display_name']}!")

            # Clear query parameters to avoid reuse of the auth code
            st.experimental_set_query_params()

            return True
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False
    else:
        # If no code in URL, display login button
        auth_url = sp_oauth.get_authorize_url()
        button_html = f"""
        <a href="{auth_url}" target="_self" style="text-decoration: none;">
            <button style="
                background-color: #1DB954;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-size: 16px;
                cursor: pointer;
            ">
                Login with Spotify
            </button>
        </a>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        return False

def logout_user():
    # Clear session state keys
    for key in ["spotify_token", "spotify_client", "user_info"]:
        if key in st.session_state:
            del st.session_state[key]

    # Remove the local Spotipy cache file so that a new login is forced
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    st.success("Logged out successfully!")
