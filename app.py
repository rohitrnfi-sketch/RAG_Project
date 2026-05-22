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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── App Background ── */
.stApp { background-color: #212121 !important; }
.main  { background-color: #212121 !important; color: #ececec; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2e2e2e !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #8a8a8a !important;
    font-size: 12px !important;
}

/* ── Title area ── */
.hero-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #ececec;
    padding: 4px 0;
}
.hero-sub {
    font-size: 0.8rem;
    color: #8a8a8a;
    margin-bottom: 1rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #2e2e2e;
    margin: 1rem 0;
}

/* ── Chat bubbles ── */
.user-label {
    font-size: 11px;
    color: #8a8a8a;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    text-align: right;
    margin-bottom: 4px;
}
.user-bubble {
    background: rgba(16, 163, 127, 0.1);
    border: 1px solid rgba(16, 163, 127, 0.25);
    border-radius: 12px 4px 12px 12px;
    padding: 12px 16px;
    margin: 4px 0 12px 20%;
    color: #ececec;
    font-size: 0.9rem;
    line-height: 1.65;
}

.ai-label {
    font-size: 11px;
    color: #8a8a8a;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.ai-bubble {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px;
    margin: 4px 0 12px 0;
    margin-right: 20%;
    color: #ececec;
    font-size: 0.9rem;
    line-height: 1.65;
}

/* ── File chips ── */
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    color: #ececec;
    margin: 3px 2px;
}
.file-chip::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10a37f;
}

/* ── Status badge ── */
.status-ready {
    background: rgba(16, 163, 127, 0.1);
    border: 1px solid rgba(16, 163, 127, 0.3);
    border-radius: 8px;
    padding: 8px 14px;
    color: #10a37f;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    color: #ececec !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #10a37f !important;
    box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #5a5a5a !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    color: #ececec !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #10a37f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 9px 20px !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    background: #10a37f !important;
}

/* ── Spinner & alerts ── */
.stSpinner { color: #10a37f !important; }
.stAlert   { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0


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
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen-qwq-32b",
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
        st.markdown(
            f"<div class='user-label'>Aap</div>"
            f"<div class='user-bubble'>{chat['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='ai-label'>🧠 AI</div>"
            f"<div class='ai-bubble'>{chat['content']}</div>",
            unsafe_allow_html=True
        )
        # ✅ Source chunks removed — user ko nahi dikhega

# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.qa_chain:
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Apna sawaal likho",
            placeholder="e.g. Is file mein revenue kitna hai? / What is the main topic?",
            label_visibility="collapsed",
            key=f"query_input_{st.session_state.input_counter}"  # ✅ counter se field clear hoti hai
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

        st.session_state.input_counter += 1  # ✅ counter badhao = input field empty ho jaata hai
        st.rerun()
else:
    st.info("👈 Sidebar mein **Groq API Key** dalo, **files upload** karo aur **Process Files** click karo!")