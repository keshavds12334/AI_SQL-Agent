import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI SQL Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0f1a; color: #e2e8f0; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1040 50%, #0d1b2a 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px; padding: 2.2rem 2.8rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700; line-height: 1.1;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #64748b; font-size: 0.9rem; margin-top: 6px; letter-spacing: 0.04em; }

.arch-flow {
    display: flex; align-items: center; gap: 6px;
    flex-wrap: wrap; margin-top: 14px;
}
.arch-node {
    background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3);
    border-radius: 8px; padding: 4px 12px; font-size: 11px;
    color: #a78bfa; font-weight: 500;
}
.arch-arrow { color: #475569; font-size: 14px; }

.chat-user {
    background: linear-gradient(135deg, rgba(96,165,250,0.1), rgba(139,92,246,0.08));
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px; margin: 8px 0; margin-left: 20%;
    font-size: 14px; color: #e2e8f0;
}
.chat-ai {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px; margin: 8px 0; margin-right: 5%;
    font-size: 14px;
}
.chat-ai-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 10px; font-size: 12px; color: #64748b;
}
.sql-block {
    background: #0d1117; border: 1px solid rgba(139,92,246,0.25);
    border-radius: 10px; padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #a78bfa; margin: 10px 0; line-height: 1.7;
    white-space: pre-wrap; word-break: break-all;
}
.answer-box {
    background: linear-gradient(135deg, rgba(52,211,153,0.06), rgba(96,165,250,0.04));
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 10px; padding: 12px 16px; margin: 10px 0;
    font-size: 14px; color: #34d399; font-weight: 500;
}
.kpi-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
}
.kpi-val { font-size: 1.6rem; font-weight: 700; color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
.kpi-lab { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

.schema-pill {
    display: inline-block; background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.2); border-radius: 6px;
    padding: 3px 10px; font-size: 11px; color: #60a5fa;
    font-family: 'JetBrains Mono', monospace; margin: 2px;
}
.upload-zone {
    background: rgba(139,92,246,0.04); border: 2px dashed rgba(139,92,246,0.25);
    border-radius: 16px; padding: 2.5rem; text-align: center;
}

div[data-testid="stSidebar"] { background: #080c14; border-right: 1px solid rgba(139,92,246,0.1); }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.6rem 1.5rem !important;
}
.stTextInput > div > div > input {
    background: #0d1117 !important; border: 1px solid rgba(139,92,246,0.3) !important;
    color: #e2e8f0 !important; border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
if 'db_conn'    not in st.session_state: st.session_state.db_conn    = None
if 'df'         not in st.session_state: st.session_state.df         = None
if 'table_name' not in st.session_state: st.session_state.table_name = None
if 'chat_hist'  not in st.session_state: st.session_state.chat_hist  = []
if 'schema'     not in st.session_state: st.session_state.schema     = None
if 'groq_key'   not in st.session_state: st.session_state.groq_key   = st.secrets.get("GROQ_API_KEY", "")

# ── Groq LLM Call ─────────────────────────────────────────────────
def call_groq(messages: list, api_key: str, model="llama-3.1-8b-instant") -> str:
    """Call Groq API directly via requests."""
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 1024, "temperature": 0.1}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            try:
                err_body = r.json()
                err_msg = err_body.get("error", {}).get("message", r.text)
            except:
                err_msg = r.text
            return f"ERROR {r.status_code}: {err_msg}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"

# ── CSV → SQLite ───────────────────────────────────────────────────
def load_csv_to_sqlite(df: pd.DataFrame, table_name: str):
    """Load DataFrame into in-memory SQLite and return connection."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return conn

# ── Schema Extractor ──────────────────────────────────────────────
def get_schema(conn, table_name: str) -> str:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    schema = f"Table: {table_name}\nColumns:\n"
    for col in cols:
        schema += f"  - {col[1]} ({col[2]})\n"
    return schema

# ── NL → SQL → Execute ────────────────────────────────────────────
def nl_to_sql_and_execute(question: str, schema: str, conn, table_name: str, api_key: str):
    """Convert natural language to SQL using Groq LLM, execute it, return results."""

    system_prompt = f"""You are an expert SQL analyst. Given a SQLite database schema, 
convert the user's question into a valid SQLite SQL query.

{schema}

Rules:
- Return ONLY the SQL query, nothing else
- No markdown, no explanation, no backticks
- Use the exact table name: {table_name}
- Use SQLite syntax only
- For text searches use LIKE '%value%'
- Always add LIMIT 100 unless user asks for all
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question}
    ]

    sql_query = call_groq(messages, api_key, model)

    # Clean up any accidental markdown
    sql_query = re.sub(r'```sql|```', '', sql_query).strip()

    if sql_query.startswith("ERROR"):
        return None, sql_query, None

    try:
        result_df = pd.read_sql_query(sql_query, conn)

        # Ask LLM to interpret the result
        interp_messages = [
            {"role": "system", "content": "You are a data analyst. Given a question and SQL result, give a clear, concise 1-2 sentence answer. Be specific with numbers."},
            {"role": "user",   "content": f"Question: {question}\n\nSQL Result (first 5 rows):\n{result_df.head().to_string()}\n\nTotal rows returned: {len(result_df)}"}
        ]
        interpretation = call_groq(interp_messages, api_key, model)
        return result_df, sql_query, interpretation

    except Exception as e:
        return None, sql_query, f"SQL Error: {str(e)}"

# ── Auto Chart Generator ───────────────────────────────────────────
def auto_chart(df: pd.DataFrame, question: str):
    """Automatically pick the best chart type based on data shape."""
    if df is None or df.empty or len(df.columns) < 2:
        return None

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    n_rows    = len(df)

    try:
        # Bar chart: categorical x numeric
        if cat_cols and num_cols and n_rows <= 50:
            fig = px.bar(df.head(20), x=cat_cols[0], y=num_cols[0],
                         color_discrete_sequence=["#7c3aed"],
                         template="plotly_dark",
                         title=f"{num_cols[0]} by {cat_cols[0]}")
            fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                              font_color="#e2e8f0", title_font_size=13)
            return fig

        # Line chart: time series or sequential numeric
        elif len(num_cols) >= 2:
            fig = px.scatter(df.head(100), x=num_cols[0], y=num_cols[1],
                             color_discrete_sequence=["#60a5fa"],
                             template="plotly_dark",
                             title=f"{num_cols[1]} vs {num_cols[0]}")
            fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                              font_color="#e2e8f0", title_font_size=13)
            return fig

        # Pie chart: single category count
        elif cat_cols and n_rows <= 20:
            counts = df[cat_cols[0]].value_counts().head(8)
            fig = px.pie(values=counts.values, names=counts.index,
                         color_discrete_sequence=px.colors.sequential.Purp,
                         template="plotly_dark",
                         title=f"Distribution of {cat_cols[0]}")
            fig.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0")
            return fig
    except:
        pass
    return None

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    groq_key = st.text_input("🔑 Groq API Key",
                              value=st.session_state.groq_key,
                              type="password",
                              placeholder="gsk_...",
                              help="Get free key at console.groq.com")
    if groq_key:
        st.session_state.groq_key = groq_key
    if st.session_state.groq_key:
        st.markdown('<div style="color:#34d399;font-size:12px">✅ API key loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#f87171;font-size:12px">⚠️ Add GROQ_API_KEY to Streamlit Secrets</div>', unsafe_allow_html=True)

    model = st.selectbox("🧠 LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

    st.markdown("---")
    st.markdown("### 📊 Sample Questions")
    samples = [
        "Show top 5 rows",
        "How many records are there?",
        "What is the average of each numeric column?",
        "Show the distribution of categories",
        "Which row has the maximum value?",
        "Count rows grouped by category",
    ]
    for s in samples:
        if st.button(s, key=f"samp_{s}", use_container_width=True):
            st.session_state['prefill_q'] = s

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#334155; line-height:1.8">
    <b style="color:#475569">How it works:</b><br>
    1. Upload any CSV file<br>
    2. CSV → SQLite database<br>
    3. Ask questions in plain English<br>
    4. Groq LLM → SQL query<br>
    5. Execute → Answer + Chart
    </div>
    """, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🤖 AI SQL Data Analyst Agent</div>
    <div class="hero-sub">CSV → SQLite → Natural Language → SQL → Insights · Powered by Groq LLaMA 3</div>
    <div class="arch-flow">
        <span class="arch-node">📄 CSV Upload</span>
        <span class="arch-arrow">→</span>
        <span class="arch-node">🗄️ SQLite DB</span>
        <span class="arch-arrow">→</span>
        <span class="arch-node">💬 Your Question</span>
        <span class="arch-arrow">→</span>
        <span class="arch-node">🧠 Groq LLaMA 3</span>
        <span class="arch-arrow">→</span>
        <span class="arch-node">📝 SQL Query</span>
        <span class="arch-arrow">→</span>
        <span class="arch-node">📊 Answer + Chart</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD ───────────────────────────────────────────────────
col_up, col_info = st.columns([1.2, 1])

with col_up:
    st.markdown("#### 📂 Upload CSV File")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded.name.replace('.csv', ''))
            conn = load_csv_to_sqlite(df, table_name)
            schema = get_schema(conn, table_name)

            st.session_state.df         = df
            st.session_state.db_conn    = conn
            st.session_state.table_name = table_name
            st.session_state.schema     = schema

            st.markdown(f'<div style="color:#34d399; font-size:13px; margin-top:8px">✅ Loaded <b>{uploaded.name}</b> → SQLite table <code style="color:#a78bfa">{table_name}</code></div>', unsafe_allow_html=True)

            # Show KPIs
            num_cols = df.select_dtypes(include='number').columns
            c1,c2,c3,c4 = st.columns(4)
            for col, val, lab in zip([c1,c2,c3,c4],
                [f"{len(df):,}", str(len(df.columns)),
                 str(len(num_cols)), f"{df.isnull().sum().sum():,}"],
                ["Rows","Columns","Numeric","Missing"]):
                with col:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lab">{lab}</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    else:
        st.markdown("""<div class="upload-zone">
            <div style="font-size:2.5rem">📄</div>
            <div style="color:#64748b; margin-top:8px; font-size:0.9rem">
                Drop your CSV file here<br>
                <small style="color:#334155">Any CSV with headers works</small>
            </div>
        </div>""", unsafe_allow_html=True)

with col_info:
    if st.session_state.schema:
        st.markdown("#### 🗄️ Database Schema")
        st.markdown(f'<div class="sql-block">{st.session_state.schema}</div>', unsafe_allow_html=True)

        st.markdown("#### 📋 Column Types")
        if st.session_state.df is not None:
            df = st.session_state.df
            for col in df.columns[:12]:
                dtype = str(df[col].dtype)
                color = "#a78bfa" if "int" in dtype or "float" in dtype else "#60a5fa"
                st.markdown(f'<span class="schema-pill" style="color:{color}">{col} <span style="opacity:0.5">({dtype})</span></span>', unsafe_allow_html=True)

# ── DATA PREVIEW ──────────────────────────────────────────────────
if st.session_state.df is not None:
    with st.expander("👁️ Preview Data (first 10 rows)", expanded=False):
        st.dataframe(st.session_state.df.head(10),
                     use_container_width=True,
                     hide_index=True)

# ── CHAT INTERFACE ────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💬 Ask Questions About Your Data")

# Render chat history
for chat in st.session_state.chat_hist:
    st.markdown(f'<div class="chat-user">🧑 {chat["question"]}</div>', unsafe_allow_html=True)

    result_html = f"""<div class="chat-ai">
        <div class="chat-ai-header">🤖 <b>AI SQL Agent</b> · Groq LLaMA 3</div>"""

    if chat.get("sql"):
        result_html += f'<div style="font-size:11px;color:#475569;margin-bottom:4px">Generated SQL:</div>'
        result_html += f'<div class="sql-block">{chat["sql"]}</div>'

    if chat.get("answer"):
        result_html += f'<div class="answer-box">💡 {chat["answer"]}</div>'

    result_html += "</div>"
    st.markdown(result_html, unsafe_allow_html=True)

    if chat.get("chart"):
        st.plotly_chart(chat["chart"], use_container_width=True)

    if chat.get("result_df") is not None and not chat["result_df"].empty:
        with st.expander("📊 Full Result Table"):
            st.dataframe(chat["result_df"], use_container_width=True, hide_index=True)

# ── QUESTION INPUT ────────────────────────────────────────────────
prefill = st.session_state.pop('prefill_q', '')
question = st.text_input(
    "Type your question...",
    value=prefill,
    placeholder="e.g. What is the average salary by department?",
    label_visibility="collapsed"
)

col_ask, col_clear = st.columns([3, 1])
with col_ask:
    ask_btn = st.button("🔍 Ask AI Agent", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_hist = []
        st.rerun()

# ── PROCESS QUESTION ──────────────────────────────────────────────
if ask_btn and question:
    if not st.session_state.groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
    elif st.session_state.db_conn is None:
        st.error("⚠️ Please upload a CSV file first.")
    else:
        with st.spinner("🧠 Thinking... Converting to SQL and executing..."):
            result_df, sql_query, answer = nl_to_sql_and_execute(
                question,
                st.session_state.schema,
                st.session_state.db_conn,
                st.session_state.table_name,
                st.session_state.groq_key
            )
            chart = auto_chart(result_df, question) if result_df is not None else None

        st.session_state.chat_hist.append({
            "question":  question,
            "sql":       sql_query,
            "answer":    answer,
            "result_df": result_df,
            "chart":     chart
        })
        st.rerun()

# ── FOOTER ────────────────────────────────────────────────────────
if not st.session_state.chat_hist and st.session_state.df is None:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:2rem; color:#334155">
        <div style="font-size:2rem; margin-bottom:8px">🚀</div>
        <div style="font-size:14px; font-weight:500; color:#475569">Get started</div>
        <div style="font-size:12px; margin-top:4px">1. Enter your Groq API key in the sidebar &nbsp;·&nbsp; 2. Upload a CSV &nbsp;·&nbsp; 3. Ask anything</div>
        <div style="font-size:11px; margin-top:12px; color:#1e293b">Get a free Groq API key at <span style="color:#7c3aed">console.groq.com</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="color:#1e293b; font-size:0.75rem; text-align:center;">Final Project Task 1 · AI SQL Agent · LangChain + Groq LLaMA 3 + SQLite + Streamlit</p>', unsafe_allow_html=True)
