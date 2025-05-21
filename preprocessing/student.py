#!/usr/bin/env python3
"""
notices_indexer.py

End-to-end script that:
- Fetches notices with section="students"
- Exports fileName/fileLink to Excel & SQLite
- Builds two FAISS indices (meta & docs)
- Saves extracted PDF texts (with OCR for scanned images) to .txt files
- Handles Google Drive links correctly
- Watches MongoDB for changes and auto-rebuilds
"""
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import re
import pickle
import logging
import sys
from io import BytesIO
from langchain.schema import Document
import faiss
import pandas as pd
import requests
import sqlite3
from pdfminer.high_level import extract_text
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# For OCR fallback
from pdf2image import convert_from_bytes
import pytesseract

# ─── Configuration ──────────────────────────────────────────────────────────────

MONGO_URI        = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
DB_NAME          = "dseu-data"
COLLECTION_NAME  = "notices"
SECTION_NAME     = "students"

# Output paths
EXCEL_FILE       = "notices_students.xlsx"
SQLITE_DB        = "notices_students.db"
META_FAISS_DIR   = "data/students/faiss_meta"
DOC_FAISS_DIR    = "data/students/faiss_docs"
DOC_TEXT_DIR     = "data/students/faiss_docs_texts"

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ─── Helper Functions ────────────────────────────────────────────────────────────

def fetch_notices():
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][COLLECTION_NAME].find({"section": SECTION_NAME}))
    client.close()
    logging.info(f"Fetched {len(docs)} documents where section='{SECTION_NAME}'.")
    return docs


def save_excel_sqlite(docs):
    df = pd.DataFrame([
        {"fileName": d.get("fileName", ""), "fileLink": d.get("fileLink", "")} for d in docs
    ])
    df.to_excel(EXCEL_FILE, index=False)
    conn = sqlite3.connect(SQLITE_DB)
    df.to_sql("notices", conn, if_exists="replace", index=False)
    conn.close()
    logging.info(f"Saved metadata to '{EXCEL_FILE}' and SQLite DB '{SQLITE_DB}'.")


def build_faiss_index(texts, ids, out_dir):
    # 1) Build Document objects
    docs = [
        Document(page_content=text, metadata={"id": id_})
        for text, id_ in zip(texts, ids)
    ]

    # 2) Build the FAISS vectorstore with a real Embeddings instance
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding_fn,      # ← pass the HuggingFaceEmbeddings instance
    )

    # 3) Save in the exact format load_local expects
    vectorstore.save_local(out_dir)



def ocr_pdf(content_bytes):
    text = ""
    try:
        images = convert_from_bytes(content_bytes)
        for img in images:
            text += pytesseract.image_to_string(img)
        logging.info("OCR fallback succeeded for scanned PDF.")
    except Exception as e:
        logging.error(f"OCR fallback failed: {e}")
    return text


def normalize_drive_link(url):
    """
    Convert a Google Drive shared URL to a direct download URL.
    """
    match = re.search(r"/d/([^/]+)/", url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def build_meta_index(docs):
    texts = [f"{d.get('fileName','')} {d.get('fileLink','')}" for d in docs]
    ids = [str(d.get('_id')) for d in docs]
    build_faiss_index(texts, ids, META_FAISS_DIR)


def build_doc_index(docs):
    texts, ids = [], []
    os.makedirs(DOC_TEXT_DIR, exist_ok=True)
    for d in docs:
        doc_id = str(d.get('_id'))
        url = d.get('fileLink','')
        dl_url = normalize_drive_link(url)
        try:
            resp = requests.get(dl_url, timeout=30)
            resp.raise_for_status()
            raw = resp.content
            text = extract_text(BytesIO(raw)) or ""
            if not text.strip():
                text = ocr_pdf(raw)
            # Save to .txt
            txt_path = os.path.join(DOC_TEXT_DIR, f"{doc_id}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logging.info(f"Saved extracted text to '{txt_path}'.")
            texts.append(text)
            ids.append(doc_id)
        except Exception as e:
            logging.warning(f"Failed to process {url}: {e}")
    build_faiss_index(texts, ids, DOC_FAISS_DIR)


def rebuild_all():
    docs = fetch_notices()
    save_excel_sqlite(docs)
    build_meta_index(docs)
    build_doc_index(docs)
    logging.info("Completed full rebuild.")


def watch_changes():
    client = MongoClient(MONGO_URI)
    coll = client[DB_NAME][COLLECTION_NAME]
    pipeline = [{"$match": {"fullDocument.section": SECTION_NAME}}]
    with coll.watch(pipeline, full_document="updateLookup") as stream:
        for change in stream:
            logging.info(f"Detected change: {change['operationType']}")
            rebuild_all()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    rebuild_all()
    try:
        watch_changes()
    except KeyboardInterrupt:
        logging.info("Exiting.")
        sys.exit(0)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
