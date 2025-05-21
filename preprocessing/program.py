import os
import time
import sqlite3
import pickle

import faiss
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
MONGO_URI   = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
DB_NAME     = "dseu-data"
PROG_COLL        = "programs"
DEPT_COLL        = "departments"

SQLITE_PATH      = "programs.db"
EXCEL_PATH       = "programs.xlsx"
FAISS_DIR        = "data/program_faiss"                # will contain index.faiss & index.pkl
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"      # or your choice
# ──────────────────────────────────────────────────────────────────────────────

# ensure output directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

# Mongo setup
client   = MongoClient(MONGO_URI)
db       = client[DB_NAME]
prog_c   = db[PROG_COLL]
dept_c   = db[DEPT_COLL]

from langchain_huggingface import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def rebuild_sqlite_faiss_excel():
    # 1) aggregate with department lookup
    pipeline = [
        {"$lookup": {
            "from": DEPT_COLL,
            "localField": "department",
            "foreignField": "_id",
            "as": "dept_docs"
        }},
        {"$unwind": {"path": "$dept_docs", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id": 1, "name": 1, "programLevel": 1,
            "duration": 1, "mode": 1,
            "exit_options": 1, "years": 1,
            "department_name": "$dept_docs.name"
        }}
    ]
    docs = list(prog_c.aggregate(pipeline))

    # 2) flatten into DataFrame
    rows = []
    for d in docs:
        base = {
            "_id": str(d["_id"]),
            "name": d.get("name", ""),
            "programLevel": d.get("programLevel", ""),
            "duration": d.get("duration", ""),
            "mode": d.get("mode", ""),
            "department": d.get("department_name", ""),
            "exit_options": ";".join(d.get("exit_options", []))
        }
        yrs = d.get("years", {})
        for i in range(1, 5):
            y = yrs.get(f"year{i}", {})
            base[f"year{i}_credit"]   = y.get("year_credit_text", "")
            base[f"year{i}_exit"]     = y.get("year_exit_text", "")
            base[f"year{i}_syllabus"] = y.get(f"year{i}_syllabus_link", "")
        rows.append(base)

    df = pd.DataFrame(rows)

    # 3) write to SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    df.to_sql("programs", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[{time.ctime()}] Wrote {len(df)} records into SQLite at {SQLITE_PATH}")

    # 4) write to Excel
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
    print(f"[{time.ctime()}] Wrote {len(df)} records into Excel at {EXCEL_PATH}")

    # 5) build and save LangChain FAISS vectorstore
    documents = [
        Document(page_content=row["name"], metadata=row.to_dict())
        for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embedder)
    vectorstore.save_local(FAISS_DIR)
    print(f"[{time.ctime()}] Built and saved vectorstore → {FAISS_DIR}/")




def watch_and_rebuild():
    # initial build
    rebuild_sqlite_faiss_excel()

    with prog_c.watch() as stream:
        print(f"[{time.ctime()}] Watching for changes in `{PROG_COLL}`…")
        for change in stream:
            op = change["operationType"]
            print(f"[{time.ctime()}] Change detected: {op}")
            rebuild_sqlite_faiss_excel()


if __name__ == "__main__":
    watch_and_rebuild()