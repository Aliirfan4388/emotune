import streamlit as st
from mainwork import modulework

def main():
    # Make sure the user is authenticated before showing playlist creation
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        st.warning("You need to log in first!")
        st.stop()

    st.title("Playlist Page")
    modulework()

if __name__ == "__main__":
    main()
