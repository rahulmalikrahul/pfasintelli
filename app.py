import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io

# --- FIX: New Import Path for 2026 ---
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

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
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=api_key, temperature=0)
    search = DuckDuckGoSearchRun()
    tools = [search]
    prompt = hub.pull("hwchase17/react")
    
    # Using the classic agent constructor to ensure compatibility
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Sentinel Controls")
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Enter Groq API Key:", type="password")
    states = st.multiselect("Monitoring States", ["CA", "MT", "TX", "NY"], default=["CA", "MT"])

# --- MAIN DASHBOARD ---
st.title("🛡️ PFAS Sentinel: 2026 Compliance Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("EPA Limit", "4.0 ppt")
col2.metric("Sectors", "Mining/Energy")
col3.metric("Status", "Live Monitoring")

tab_map, tab_ai = st.tabs(["🗺️ Regional Risk Map", "💬 AI Research Agent"])

with tab_map:
    map_df = pd.DataFrame({'lat': [35.3, 46.0], 'lon': [-119.0, -112.5], 'ppt': [145, 52]})
    fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="ppt", color="ppt",
                            mapbox_style="carto-darkmatter", zoom=3, height=500)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

with tab_ai:
    if groq_key:
        executor = get_pfas_agent(groq_key)
        if prompt := st.chat_input("Analyze PFAS rules for..."):
            st.chat_message("user").write(prompt)
            with st.spinner("Searching state records..."):
                res = executor.invoke({"input": prompt + f" Focus on states: {states}"})
                st.chat_message("assistant").write(res["output"])
    else:
        st.error("Missing API Key in Sidebar or Secrets.")