import streamlit as st
from utils.api_client import get_health_status

st.markdown(
    """
<style>
    .hero-container {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #1A5276 0%, #2980B9 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .nav-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        border-color: #3498db;
    }
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        color: #7f8c8d;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Hero Section ---
st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">Hydro AI Agent</div>
    <div class="hero-subtitle">
        Simple flood susceptibility and rainfall prediction system for the Mahanadi Basin
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# --- System Status ---
status = get_health_status()
if status:
    st.success("✅ **System Status:** Backend API is online and connected.", icon="🟢")
else:
    st.error(
        "❌ **System Status:** Backend API is currently offline. Please start FastAPI.",
        icon="🔴",
    )

st.markdown("<br>", unsafe_allow_html=True)


# --- Navigation Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    <div class="nav-card">
        <div class="card-icon">🌊</div>
        <div class="card-title">Flood Susceptibility</div>
        <div class="card-desc">Geospatial risk assessment using XGBoost and raster features.</div>
        <div style="margin-top: 1rem; display: inline-block; background: #e8f4f8; color: #3498db; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
            API Connected ✅
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Open Flood Model", key="btn_flood", width="stretch"):
        st.switch_page("pages/flood_page.py")

with col2:
    st.markdown(
        """
    <div class="nav-card">
        <div class="card-icon">🌧️</div>
        <div class="card-title">Rainfall Prediction</div>
        <div class="card-desc">3-day forecast with complete basin, search-by-location, and map-wise modes.</div>
        <div style="margin-top: 1rem; display: inline-block; background: #e8f4f8; color: #3498db; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
            API Connected ✅
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Open Rainfall Model", key="btn_rain", width="stretch"):
        st.switch_page("pages/rainfall_page.py")

with col3:
    st.markdown(
        """
    <div class="nav-card">
        <div class="card-icon">💬</div>
        <div class="card-title">AI Chat Assistant</div>
        <div class="card-desc">Ask natural language questions to the LangChain hydrology agent.</div>
        <div style="margin-top: 1rem; display: inline-block; background: #eaeded; color: #7f8c8d; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
            Agent Available 🤖
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Open AI Chat", key="btn_chat", width="stretch"):
        st.switch_page("pages/chat_page.py")


# --- Footer ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "Built using Streamlit, FastAPI, and ML models. Part of the Hydro AI student project."
)
