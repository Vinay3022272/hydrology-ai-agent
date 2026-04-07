import sys
import os
import streamlit as st

# --- Ensure existing flood app can import its local modules ---
FLOOD_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "Flood_Model", "flood_app"))
if FLOOD_APP_DIR not in sys.path:
    sys.path.insert(0, FLOOD_APP_DIR)

# --- Global Page Configuration ---
st.set_page_config(
    page_title="Hydro AI Agent",
    page_icon="🌊",
    layout="wide",
)

st.logo("https://cdn-icons-png.flaticon.com/512/3067/3067184.png") # Optional water icon

# --- Global CSS ---
st.markdown("""
<style>
    /* Styling to make sidebar clean */
    [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Define Pages ---
home_page = st.Page("pages/home_page.py", title="Home", icon="🏠", default=True)
flood_page = st.Page("pages/flood_page.py", title="Flood Susceptibility", icon="🌊")
rainfall_page = st.Page("pages/rainfall_page.py", title="Rainfall Prediction", icon="🌧️")
chat_page = st.Page("pages/chat_page.py", title="AI Chat Assistant", icon="💬")


# --- Navigation Setup ---
pg = st.navigation(
    {
        "Overview": [home_page],
        "Hydrology Models": [flood_page, rainfall_page],
        "Assistant": [chat_page],
    }
)

# Run chosen page
pg.run()
