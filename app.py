import streamlit as st
from auth import authenticate_user, logout_user
import warnings
from dotenv import load_dotenv
load_dotenv()


# Hide all DeprecationWarnings (not recommended for production)
warnings.filterwarnings("ignore")

def main():
    st.set_page_config(page_title="Spotify Playlist Creator", page_icon="🎵")
    st.title("Emotune")

    # Initialize session state variables if they don't exist
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "spotify_client" not in st.session_state:
        st.session_state.spotify_client = None
    if "user_info" not in st.session_state:
        st.session_state.user_info = None

    # If not logged in, show the login flow
    if not st.session_state.authenticated:
        logged_in = authenticate_user()
        if logged_in:
            # Mark user as authenticated and set query_params to redirect to 'Playlist'
            st.session_state.authenticated = True
            st.experimental_set_query_params(page="Playlist")
            st.experimental_rerun()  # Forces a rerun to update the page
    else:
        # If already logged in, display user info
        if st.session_state.get("user_info"):
            st.success(f"Already logged in as {st.session_state.user_info['display_name']}.")
        else:
            st.info("User info not found.")

        # Provide a logout button
        if st.button("Logout"):
            logout_user()
            st.session_state.authenticated = False  # Reset the auth flag
            st.experimental_rerun()  # Rerun the app to show the login flow again

if __name__ == "__main__":
    main()
