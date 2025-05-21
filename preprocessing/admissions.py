#!/usr/bin/env python3
"""
notices_indexer.py

End-to-end script that:
- Fetches notices with section="admissions"
- Exports fileName/fileLink to Excel & SQLite
- Builds two FAISS indices (meta & docs)
- Saves extracted PDF texts (with OCR for scanned images) to .txt files
- Handles Google Drive links correctly
- Watches MongoDB for changes and auto-rebuilds
"""

import os
import re
import pickle
import logging
import sys
from io import BytesIO

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
SECTION_NAME     = "admission"

# Output paths
EXCEL_FILE       = "notices_admissions.xlsx"
SQLITE_DB        = "notices_admissions.db"
META_FAISS_DIR   = "data/admissions/faiss_meta"
DOC_FAISS_DIR    = "data/admissions/faiss_docs"
DOC_TEXT_DIR     = "admissions/faiss_docs_texts"

# Embedding model
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

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


from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

def build_faiss_index(texts, ids, out_dir):
    if not texts:
        logging.warning(f"No texts to index for '{out_dir}', skipping.")
        return

    # 1) Compute embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # 2) Build the FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 3) Persist the index file
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))

    # 4) Build & pickle the docstore + id map (as before)
    docs = [Document(page_content=t, metadata={}) for t in texts]
    docstore = InMemoryDocstore({id_: doc for id_, doc in zip(ids, docs)})
    index_to_docstore_id = {i: id_ for i, id_ in enumerate(ids)}
    with open(os.path.join(out_dir, "index.pkl"), "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    logging.info(f"Built FAISS index in '{out_dir}/index.faiss'.")


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
























# """
# notices_indexer.py

# - Fetches notices with section="admission"
# - Exports metadata to Excel & SQLite
# - Builds two FAISS indices:
#     • meta (fileName + fileLink)
#     • docs (full extracted text)
# - OCR fallback for scanned PDFs
# - Watches MongoDB for changes and auto-rebuilds
# """

# import os
# import re
# import sys
# import pickle
# import logging
# import sqlite3
# from io import BytesIO

# import faiss
# import pandas as pd
# import requests
# from pdfminer.high_level import extract_text
# from pdf2image import convert_from_bytes
# import pytesseract
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer

# # ─── CONFIG ────────────────────────────────────────────────────────────────────
# MONGO_URI       = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
# DB_NAME         = "dseu-data"
# COLLECTION_NAME = "notices"
# SECTION_NAME    = "admission"

# EXCEL_FILE      = "notices_admissions.xlsx"
# SQLITE_DB       = "notices_admissions.db"
# META_FAISS_DIR  = "data/admissions/faiss_meta"
# DOC_FAISS_DIR   = "data/admissions/faiss_docs"
# DOC_TEXT_DIR    = "data/admissions/faiss_docs_texts"

# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# def fetch_notices():
#     client = MongoClient(MONGO_URI)
#     docs = list(client[DB_NAME][COLLECTION_NAME].find({"section": SECTION_NAME}))
#     client.close()
#     logging.info(f"Fetched {len(docs)} '{SECTION_NAME}' documents.")
#     return docs


# def save_excel_sqlite(docs):
#     df = pd.DataFrame([
#         {"fileName": d.get("fileName", ""), "fileLink": d.get("fileLink", "")}
#         for d in docs
#     ])
#     df.to_excel(EXCEL_FILE, index=False)
#     conn = sqlite3.connect(SQLITE_DB)
#     df.to_sql("notices", conn, if_exists="replace", index=False)
#     conn.close()
#     logging.info(f"Saved metadata to '{EXCEL_FILE}' and '{SQLITE_DB}'.")


# def build_faiss_index(texts, ids, out_dir):
#     if not texts:
#         logging.warning(f"No texts for indexing in '{out_dir}'.")
#         return
#     model = SentenceTransformer(EMBEDDING_MODEL)
#     embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     os.makedirs(out_dir, exist_ok=True)
#     faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
#     with open(os.path.join(out_dir, "index.pkl"), "wb") as f:
#         pickle.dump(ids, f)
#     logging.info(f"Built FAISS index at '{out_dir}/index.faiss'.")


# def ocr_pdf(content_bytes):
#     try:
#         text = ""
#         images = convert_from_bytes(content_bytes)
#         for img in images:
#             text += pytesseract.image_to_string(img)
#         logging.info("OCR fallback succeeded.")
#         return text
#     except Exception as e:
#         logging.error(f"OCR failed: {e}")
#         return ""


# def normalize_drive_link(url):
#     m = re.search(r"/d/([^/]+)/", url)
#     if m:
#         return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
#     return url


# def build_meta_index(docs):
#     texts = [f"{d.get('fileName','')} {d.get('fileLink','')}" for d in docs]
#     ids   = [str(d.get("_id")) for d in docs]
#     build_faiss_index(texts, ids, META_FAISS_DIR)


# def build_doc_index(docs):
#     texts, ids = [], []
#     os.makedirs(DOC_TEXT_DIR, exist_ok=True)

#     for d in docs:
#         doc_id = str(d.get("_id"))
#         link   = normalize_drive_link(d.get("fileLink",""))
#         try:
#             resp = requests.get(link, timeout=30)
#             resp.raise_for_status()
#             raw = resp.content
#             text = extract_text(BytesIO(raw)).strip() or ocr_pdf(raw)
#             path = os.path.join(DOC_TEXT_DIR, f"{doc_id}.txt")
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(text)
#             texts.append(text)
#             ids.append(doc_id)
#             logging.info(f"Saved text for ID {doc_id}.")
#         except Exception as e:
#             logging.warning(f"Failed to process {link}: {e}")

#     build_faiss_index(texts, ids, DOC_FAISS_DIR)


# def rebuild_all():
#     docs = fetch_notices()
#     save_excel_sqlite(docs)
#     build_meta_index(docs)
#     build_doc_index(docs)
#     logging.info("Full rebuild complete.")


# def watch_changes():
#     client = MongoClient(MONGO_URI)
#     coll = client[DB_NAME][COLLECTION_NAME]
#     pipeline = [{"$match": {"fullDocument.section": SECTION_NAME}}]
#     with coll.watch(pipeline, full_document="updateLookup") as stream:
#         for change in stream:
#             logging.info(f"Change detected: {change['operationType']}")
#             rebuild_all()


# def main():
#     logging.basicConfig(level=logging.INFO,
#                         format="%(asctime)s [%(levelname)s] %(message)s")
#     rebuild_all()
#     try:
#         watch_changes()
#     except KeyboardInterrupt:
#         logging.info("Stopping watcher.")
#         sys.exit(0)


# if __name__ == "__main__":
#     main()
