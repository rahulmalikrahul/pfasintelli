import streamlit as st
import pandas as pd
import plotly.express as px

# --- THE PERMANENT FIX: Modern LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# --- 1. UI & THEME CONFIGURATION ---
st.set_page_config(page_title="PFAS Sentinel", page_icon="💧", layout="wide")

st.markdown("""
    <style>
    /* Sleek Dark Theme & Glassmorphism */
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: white;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #00d4ff !important; 
        color: black !important; 
        font-weight: bold; 
    }
    .stChatFloatingInputContainer { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE AI ENGINE (Zero-Hub LangGraph) ---
@st.cache_resource
def setup_agent(api_key):
    if not api_key:
        return None
    try:
        # Fast inference with Llama 3 on Groq
        llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=api_key, temperature=0)
        
        # Search Tool (Replaces LegiScan)
        search = DuckDuckGoSearchRun()
        tools = [search]
        
        # Modern LangGraph Agent (Replaces legacy AgentExecutor and Hub)
        return create_react_agent(llm, tools)
    except Exception as e:
        st.error(f"Agent Initialization Error: {e}")
        return None

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3144/3144600.png", width=80)
    st.title("Sentinel Controls")
    st.divider()
    
    # Secrets handling
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Enter Groq API Key:", type="password")
        
    states = st.multiselect("Active Jurisdictions", ["CA", "MT", "TX", "NY", "WV", "PA"], default=["CA", "MT"])
    st.info("Agent Mode: Live Web-Search (LangGraph Engine)")

# --- 4. MAIN DASHBOARD INTERFACE ---
st.title("🛡️ PFAS Sentinel: Intelligence Portal")
st.caption("Real-time monitoring of industrial contamination and regulatory mandates.")

# KPI Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("EPA Target Limits", "4.0 ppt", "Federal Standard")
col2.metric("Active Monitored States", f"{len(states)}", "Live Polling")
col3.metric("System Core", "LangGraph + Llama 3", "Operational")

st.divider()

# Core Tabs
tab_map, tab_ai = st.tabs(["🗺️ Regional Risk Map", "💬 AI Intelligence Agent"])

with tab_map:
    st.subheader("Interactive Contamination Heatmap")
    
    # Visually striking dark-themed map
    map_data = pd.DataFrame({
        'lat': [35.3, 46.0, 31.9, 38.8, 41.2], 
        'lon': [-119.0, -112.5, -102.4, -80.6, -77.0], 
        'Concentration (ppt)': [145, 52, 12, 88, 34],
        'Site': ['CA Refinery', 'MT Mining Site', 'TX Facility', 'WV Plant', 'PA Industrial']
    })
    
    fig = px.scatter_mapbox(map_data, lat="lat", lon="lon", size="Concentration (ppt)", 
                            color="Concentration (ppt)", hover_name="Site",
                            color_continuous_scale="Viridis", mapbox_style="carto-darkmatter",
                            zoom=3.5, height=550)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with tab_ai:
    st.subheader("Deep Research Agent")
    st.markdown("Ask the Sentinel to query the latest environmental mandates across your selected states.")
    
    if groq_key:
        agent = setup_agent(groq_key)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input execution
        if user_input := st.chat_input("Ex: What are the current PFAS limits for Montana mines?"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Initiating web search and analyzing records..."):
                    try:
                        # LangGraph standard execution
                        context = f"Context: Focus specifically on {', '.join(states)}. "
                        inputs = {"messages": [("user", context + user_input)]}
                        response = agent.invoke(inputs)
                        
                        # Extract the final AI message
                        answer = response["messages"][-1].content
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Search failed. Error details: {e}")
    else:
        st.warning("⚠️ Enter your GROQ_API_KEY in the sidebar to activate the AI Agent.")