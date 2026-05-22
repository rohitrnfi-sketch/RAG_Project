# file_loader.py
# PDF, Word, Excel teeno file types ko text mein convert karta hai

import pandas as pd
from pypdf import PdfReader
from docx import Document


def load_pdf(file) -> str:
    """PDF se text extract karo"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_docx(file) -> str:
    """Word document se text extract karo"""
    doc = Document(file)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def load_excel(file) -> str:
    """Excel file ke saare sheets se text extract karo"""
    xl = pd.ExcelFile(file)
    all_text = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        all_text.append(f"--- Sheet: {sheet} ---")
        all_text.append(df.to_string(index=False))
    return "\n\n".join(all_text)


def load_file(uploaded_file) -> str:
    """
    Streamlit UploadedFile object ko accept karta hai aur
    uska content text mein return karta hai.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return load_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return load_docx(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return load_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")
