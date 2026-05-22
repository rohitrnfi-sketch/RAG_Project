# app.py  ──  RAG AI App with Groq + Streamlit
# Run: streamlit run app.py

import streamlit as st
from file_loader import load_file
from rag_engine import split_text, build_vectorstore, build_qa_chain, ask_question

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.main { background-color: #0f0f13; color: #e8e6e0; }
.stApp { background-color: #0f0f13; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a24 0%, #12121a 100%);
    border-right: 1px solid #2a2a3a;
}

/* Heading */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8c97e, #d4845a, #c05c7e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #2a1f3d, #1e1a2e);
    border: 1px solid #3d2d5e;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 10px 0;
    margin-left: 15%;
    color: #e8e6e0;
    font-size: 0.95rem;
}

.ai-bubble {
    background: linear-gradient(135deg, #1a1f2e, #131820);
    border: 1px solid #2a3550;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 10px 0;
    margin-right: 15%;
    color: #e8e6e0;
    font-size: 0.95rem;
    line-height: 1.6;
}

.ai-label {
    font-size: 0.75rem;
    color: #e8c97e;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: 'Syne', sans-serif;
}

.user-label {
    font-size: 0.75rem;
    color: #c05c7e;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
    text-align: right;
    font-family: 'Syne', sans-serif;
}

/* File chip */
.file-chip {
    display: inline-block;
    background: #1e2a1e;
    border: 1px solid #3a5a3a;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #7ec87e;
    margin: 3px;
}

/* Status badge */
.status-ready {
    background: #1a2e1a;
    border: 1px solid #3a7a3a;
    border-radius: 8px;
    padding: 8px 14px;
    color: #7ec87e;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Source expander */
.source-box {
    background: #141420;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #888;
    margin-top: 8px;
    font-family: monospace;
}

/* Input box */
.stTextInput > div > div > input {
    background: #1a1a24 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 12px !important;
    color: #e8e6e0 !important;
    padding: 12px 16px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #e8c97e, #d4845a) !important;
    color: #0f0f13 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    padding: 10px 24px !important;
    letter-spacing: 0.05em !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

.divider {
    border: none;
    border-top: 1px solid #2a2a3a;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="https://console.groq.com se free API key lo"
    )

    model_choice = st.selectbox(
        "🤖 Model",
        options=[
            "llama-3.3-70b-versatile",    # ✅ Best quality (recommended)
            "llama-3.1-8b-instant",        # ✅ Fast & lightweight
            "llama4-scout-17b-16e-instruct", # ✅ Llama 4 Scout
            "meta-llama/llama-4-maverick-17b-128e-instruct",  # ✅ Llama 4 Maverick
            "qwen-qwq-32b",               # ✅ Reasoning model
        ],
        index=0,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📁 Files Upload Karo")

    uploaded_files = st.file_uploader(
        "PDF, Word, Excel drag karo",
        type=["pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_btn = st.button("🚀 Process Files", use_container_width=True)

    # Already uploaded files show karo
    if st.session_state.uploaded_names:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**✅ Loaded Files:**")
        for name in st.session_state.uploaded_names:
            st.markdown(f"<span class='file-chip'>📄 {name}</span>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Made with ❤️ using Groq + LangChain + FAISS")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='hero-title'>🧠 RAG AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Files upload karo → Query poochho → AI dhundh ke jawab dega</div>", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── File Processing ────────────────────────────────────────────────────────────
if process_btn:
    if not groq_api_key:
        st.error("⚠️ Pehle sidebar mein Groq API Key dalo!")
    elif not uploaded_files:
        st.warning("⚠️ Koi file upload nahi ki!")
    else:
        with st.spinner("📖 Files padh raha hoon..."):
            all_text = ""
            names = []
            errors = []

            for f in uploaded_files:
                try:
                    text = load_file(f)
                    all_text += f"\n\n=== File: {f.name} ===\n\n" + text
                    names.append(f.name)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")

            if all_text.strip():
                with st.spinner("🔪 Text chunks mein tod raha hoon..."):
                    chunks = split_text(all_text)

                with st.spinner(f"🧲 {len(chunks)} chunks embed kar raha hoon..."):
                    vectorstore = build_vectorstore(chunks)

                with st.spinner("🤖 AI chain bana raha hoon..."):
                    st.session_state.qa_chain = build_qa_chain(
                        vectorstore, groq_api_key, model_choice
                    )
                    st.session_state.uploaded_names = names

                st.success(f"✅ {len(names)} file(s) ready! Ab koi bhi sawaal poochho.")
                st.rerun()

# ── Status ─────────────────────────────────────────────────────────────────────
if st.session_state.qa_chain:
    st.markdown(
        f"<div class='status-ready'>✅ {len(st.session_state.uploaded_names)} file(s) loaded — AI ready hai!</div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

# ── Chat History ───────────────────────────────────────────────────────────────
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-label'>Aap</div><div class='user-bubble'>{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-label'>🧠 AI</div><div class='ai-bubble'>{chat['content']}</div>", unsafe_allow_html=True)
        if chat.get("sources"):
            with st.expander("📎 Source Chunks dekho"):
                for i, doc in enumerate(chat["sources"][:3], 1):
                    st.markdown(f"<div class='source-box'><b>Chunk {i}:</b><br>{doc.page_content[:300]}...</div>", unsafe_allow_html=True)

# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.qa_chain:
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Apna sawaal likho",
            placeholder="e.g. Is file mein revenue kitna hai? / What is the main topic?",
            label_visibility="collapsed",
            key="query_input"
        )
    with col2:
        send_btn = st.button("Send ➤", use_container_width=True)

    if send_btn and user_query.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("🔍 Dhundh raha hoon..."):
            try:
                result = ask_question(st.session_state.qa_chain, user_query)
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": f"❌ Error aaya: {str(e)}",
                    "sources": [],
                })

        st.rerun()
else:
    st.info("👈 Sidebar mein **Groq API Key** dalo, **files upload** karo aur **Process Files** click karo!")