import google.generativeai as genai
from spotify_config import API_KEY_GEMINI


import re

genai.configure(api_key = API_KEY_GEMINI)

model = genai.GenerativeModel("gemini-1.5-flash")



def extract_song_titles(response_text):
    """Extracts song titles from Gemini's response text."""
    lines = response_text.strip().split('\n')
    song_titles = [line.strip().lstrip('* ') for line in lines if line.strip()]
    return song_titles

def fetchsongs():

    response = model.generate_content("Recommend some depressing songs for my playlist only 5 songs should be there dont ask any question just 10 song no extra texts also remove the serial no.")
    song_titles = extract_song_titles(response.text)
    return song_titles

