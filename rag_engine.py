# rag_engine.py
# Text ko chunks mein todta hai, embed karta hai, FAISS mein store karta hai
# aur query ke basis pe relevant chunks dhundh ke Groq se answer leta hai

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


# ── QA Chain banana (LCEL style — naye LangChain ke saath compatible) ──────────
def build_qa_chain(vectorstore, groq_api_key: str, model_name: str):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
Aap ek helpful assistant hain. Neeche diye gaye context ke basis pe user ke sawaal ka jawab do.
Agar jawab context mein nahi hai toh clearly bolo "Mujhe is file mein yeh information nahi mili."
Jawab Hindi ya English mein de sakte ho jis mein user ne poochha ho.

Context:
{context}

Sawaal: {question}

Jawab:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain — RetrievalQA ka modern replacement
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


# ── Query run karna ────────────────────────────────────────────────────────────
def ask_question(qa_chain: dict, question: str) -> dict:
    chain    = qa_chain["chain"]
    retriever = qa_chain["retriever"]

    answer  = chain.invoke(question)
    sources = retriever.invoke(question)   # source docs bhi fetch karo

    return {
        "answer": answer,
        "sources": sources,
    }