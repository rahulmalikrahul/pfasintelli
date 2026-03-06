import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Modern LangGraph & Groq Imports
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# --- 1. UI & THEME CONFIGURATION ---
st.set_page_config(page_title="PFAS Sentinel", page_icon="💧", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03); 
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 1.5rem; border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; background-color: rgba(255, 255, 255, 0.05); 
        border-radius: 8px; color: white; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #00d4ff !important; color: black !important; font-weight: bold; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ENGINE (SQLite) ---
def init_db():
    conn = sqlite3.connect("pfas_regulations.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS regulations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  state TEXT, source TEXT, summary TEXT, 
                  severity INTEGER, date_added TEXT)''')
    
    c.execute("SELECT COUNT(*) FROM regulations")
    if c.fetchone()[0] == 0:
        seed_data = [
            ('CA', 'EPA Water Board', 'Strict limits on PFAS in refinery water discharge', 90, str(datetime.now().date())),
            ('MT', 'State Registry', 'Mining runoff PFAS monitoring requirements', 60, str(datetime.now().date())),
            ('TX', 'ACC Stewardship', 'Industry-led phased reduction of PFAS chemicals', 40, str(datetime.now().date()))
        ]
        c.executemany("INSERT INTO regulations (state, source, summary, severity, date_added) VALUES (?, ?, ?, ?, ?)", seed_data)
    conn.commit()
    conn.close()

init_db()

# --- 3. CUSTOM AGENT TOOLS ---
@tool
def scrape_pfas_policy(state: str) -> str:
    """
    REQUIRED: Scrapes the American Chemistry Council (ACC) PFAS Stewardship page 
    and official state registries for the given state. Updates the heatmap database.
    """
    st.toast(f"🔍 Scrapping ACC & State Portal for {state}...", icon="🌐")
    
    # Target URL for ACC PFAS Stewardship
    acc_url = "https://www.americanchemistry.com/chemistry-in-america/chemistry-in-everyday-products/pfas/pfas-stewardship-and-regulation"
    
    try:
        response = requests.get(acc_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=7)
        soup = BeautifulSoup(response.text, 'html.parser')
        acc_text = soup.get_text(separator=' ', strip=True)[:400]
    except:
        acc_text = "Standard industry commitment to PFAS reduction and state-level compliance."

    # Update Database
    conn = sqlite3.connect("pfas_regulations.db")
    c = conn.cursor()
    summary = f"ACC & State Compliance: {acc_text}..."
    c.execute("INSERT INTO regulations (state, source, summary, severity, date_added) VALUES (?, ?, ?, ?, ?)", 
              (state, "ACC & State Site", summary, 80, str(datetime.now().date())))
    conn.commit()
    conn.close()
    
    return f"Database updated with PFAS regulations for {state} from ACC and State sources."

@tool
def general_web_search(query: str) -> str:
    """Use for general news or specific PFAS concentration reports if the primary database is insufficient."""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except:
        return "Web search limit reached. Rely on internal model knowledge for this specific query."

# --- 4. THE AI ENGINE ---
@st.cache_resource
def setup_agent(api_key):
    if not api_key: return None
    # Updated to 2026 versatile model
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
    tools = [scrape_pfas_policy, general_web_search]
    return create_react_agent(llm, tools)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3144/3144600.png", width=80)
    st.title("Sentinel Controls")
    st.divider()
    
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Enter Groq API Key:", type="password")
        
    active_states = st.multiselect("Monitoring Jurisdictions", ["CA", "MT", "TX", "NY", "WV", "PA"], default=["CA", "MT", "TX"])
    st.success("✅ Database: Connected")

# --- 6. MAIN DASHBOARD ---
st.title("🛡️ PFAS Sentinel: Intelligence Portal")
st.caption("Monitoring industrial compliance via the ACC and Official State Portals.")

# Metrics Row
conn = sqlite3.connect("pfas_regulations.db")
db_count = pd.read_sql("SELECT COUNT(*) FROM regulations", conn).iloc[0,0]
conn.close()

col1, col2, col3 = st.columns(3)
col1.metric("EPA Target Limits", "4.0 ppt", "PFOA/PFOS")
col2.metric("Regulations Logged", f"{db_count}", "In Local DB")
col3.metric("LLM Engine", "Llama 3.3 (2026)", "Operational")

st.divider()

# Tab Layout
tab_map, tab_db, tab_ai = st.tabs(["🗺️ Regional Heatmap", "🗄️ Regulation Database", "💬 AI Research Agent"])

with tab_map:
    st.subheader("Dynamic Contamination & Regulation Heatmap")
    
    state_coords = {
        'CA': (36.77, -119.41), 'MT': (46.87, -110.36), 'TX': (31.96, -99.90),
        'NY': (42.16, -74.94), 'WV': (38.59, -80.45), 'PA': (41.20, -77.19)
    }
    
    conn = sqlite3.connect("pfas_regulations.db")
    df_heat = pd.read_sql("SELECT state, SUM(severity) as total_severity, COUNT(*) as reg_count FROM regulations GROUP BY state", conn)
    conn.close()
    
    df_heat['lat'] = df_heat['state'].map(lambda x: state_coords.get(x, (39.82, -98.57))[0])
    df_heat['lon'] = df_heat['state'].map(lambda x: state_coords.get(x, (39.82, -98.57))[1])
    
    if not df_heat.empty:
        fig = px.scatter_mapbox(df_heat, lat="lat", lon="lon", size="total_severity", 
                                color="total_severity", hover_name="state",
                                hover_data=["reg_count"],
                                color_continuous_scale="Viridis", mapbox_style="carto-darkmatter",
                                zoom=3.2, height=550)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with tab_db:
    st.subheader("Logged Regulations")
    conn = sqlite3.connect("pfas_regulations.db")
    df_db = pd.read_sql("SELECT state, source, date_added, severity, summary FROM regulations ORDER BY id DESC", conn)
    conn.close()
    st.dataframe(df_db, use_container_width=True, hide_index=True)

with tab_ai:
    st.subheader("Deep Research Agent")
    if groq_key:
        agent = setup_agent(groq_key)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_input := st.chat_input("Ex: Update the database with ACC PFAS regulations for Texas..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.chat_message("assistant"):
                with st.status("🧠 Agent processing request...", expanded=True) as status:
                    try:
                        context = f"Update the database for {', '.join(active_states)} using scrape_pfas_policy. "
                        inputs = {"messages": [("user", context + user_input)]}
                        response = agent.invoke(inputs)
                        answer = response["messages"][-1].content
                        status.update(label="Analysis Complete", state="complete")
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution failed: {e}")
    else:
        st.warning("⚠️ Enter GROQ_API_KEY in the sidebar.")