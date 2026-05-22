# 🧠 RAG AI Assistant — Setup Guide

## Project Structure
```
rag_project/
├── app.py            ← Main Streamlit app
├── file_loader.py    ← PDF, Word, Excel reader
├── rag_engine.py     ← Chunking, Embedding, FAISS, Groq chain
├── requirements.txt  ← Saari dependencies
└── README.md
```

---

## ⚡ Step-by-Step Setup

### 1. Virtual Environment banao (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Dependencies install karo
```bash
pip install -r requirements.txt
```

### 3. Groq API Key lo (FREE hai!)
- https://console.groq.com pe jaao
- Account banao → API Keys → Create Key
- Key copy karo (gsk_... se shuru hogi)

### 4. App run karo
```bash
streamlit run app.py
```

---

## 🚀 App kaise use karein

1. Browser mein app khulega (http://localhost:8501)
2. **Sidebar mein Groq API Key** dalo
3. **Model select karo** (llama3-70b recommended)
4. **Files upload karo** — PDF, Word (.docx), Excel (.xlsx) sab chalega
5. **"Process Files"** button dabao
6. **Sawaal poochho** — AI file se dhundh ke jawab dega!

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| LLM | Groq (Llama3 / Mixtral) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | FAISS (local) |
| Orchestration | LangChain |
| File Parsing | pypdf, python-docx, openpyxl |

---

## 📝 Notes
- Pehli baar embedding model download hoga (~90MB) — internet chahiye
- Uske baad offline bhi kaam karega (embeddings ke liye)
- Groq API free tier mein kaafi requests milti hain
