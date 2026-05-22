# rag_engine.py
# Text ko chunks mein todta hai, embed karta hai, FAISS mein store karta hai
# aur query ke basis pe relevant chunks dhundh ke Groq se answer leta hai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st


# ── Embedding model (free, local) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ── Text → Chunks ──────────────────────────────────────────────────────────────
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


# ── Chunks → FAISS Vector Store ────────────────────────────────────────────────
def build_vectorstore(chunks: list[str]):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


# ── QA Chain banana ────────────────────────────────────────────────────────────
def build_qa_chain(vectorstore, groq_api_key: str, model_name: str):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2,
    )

    prompt_template = """
Aap ek helpful assistant hain. Neeche diye gaye context ke basis pe user ke sawaal ka jawab do.
Agar jawab context mein nahi hai toh clearly bolo "Mujhe is file mein yeh information nahi mili."
Jawab Hindi ya English mein de sakte ho jis mein user ne poochha ho.

Context:
{context}

Sawaal: {question}

Jawab:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


# ── Query run karna ────────────────────────────────────────────────────────────
def ask_question(qa_chain, question: str) -> dict:
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": result.get("source_documents", []),
    }
