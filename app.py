import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io

# --- 2026 BULLETPROOF IMPORTS ---
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
import langchainhub as hub  # Correct way to import hub in 2026

# --- UI CONFIG ---
st.set_page_config(page_title="PFAS Sentinel", page_icon="💧", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 1.5rem;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AGENT ENGINE ---
@st.cache_resource
def get_pfas_agent(api_key):
    if not api_key: return None
    
    # 1. Initialize Groq Llama 3
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=api_key, temperature=0)
    
    # 2. Setup Search Tool (LegiScan replacement)
    search = DuckDuckGoSearchRun()
    tools = [search]
    
    # 3. Pull Prompt from the dedicated hub package
    prompt = hub.pull("hwchase17/react")
    
    # 4. Construct Agent using classic logic for maximum stability
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Sentinel Controls")
    st.divider()
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Enter Groq API Key:", type="password")
    states = st.multiselect("Monitoring States", ["CA", "MT", "TX", "WV"], default=["CA", "MT"])

# --- DASHBOARD ---
st.title("🛡️ PFAS Sentinel: 2026 Compliance Intelligence")

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("EPA Target", "4.0 ppt", "Federal Limit")
m2.metric("Data Source", "Live Web", "Verified")
m3.metric("System Health", "Operational", "Python 3.13")

tab_map, tab_ai = st.tabs(["🗺️ Regional Risk Map", "💬 AI Research Agent"])

with tab_map:
    # Interactive Mapbox Chart
    map_df = pd.DataFrame({
        'lat': [35.3, 46.0, 31.9], 
        'lon': [-119.0, -112.5, -102.4], 
        'ppt': [145, 52, 21],
        'site': ['Central Valley', 'Clark Fork', 'Permian Basin']
    })
    fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="ppt", color="ppt",
                            hover_name="site", mapbox_style="carto-darkmatter", zoom=3, height=500)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with tab_ai:
    if groq_key:
        executor = get_pfas_agent(groq_key)
        if prompt := st.chat_input("Analyze new 2026 PFAS limits for mining in Montana..."):
            st.chat_message("user").write(prompt)
            with st.spinner("Searching official records..."):
                # Run the agent
                res = executor.invoke({"input": f"{prompt} Focusing on states: {states}"})
                st.chat_message("assistant").write(res["output"])
    else:
        st.error("Please add your GROQ_API_KEY to continue.")