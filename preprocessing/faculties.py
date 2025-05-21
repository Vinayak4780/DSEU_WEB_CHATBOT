# import os
# import time
# import sqlite3
# import pickle

# import faiss
# import numpy as np
# import pandas as pd
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer

# # ──────────────────────────────────────────────────────────────────────────────
# # CONFIG
# MONGO_URI        = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
# DB_NAME          = "dseu-data"
# FAC_COLL         = "faculties"
# DEPT_COLL        = "departments"

# SQLITE_PATH      = "faculties.db"
# EXCEL_PATH       = "faculties.xlsx"
# FAISS_DIR        = "data/faculties_folder"                 # will contain index.faiss & index.pkl
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"       # or any model you prefer
# # ──────────────────────────────────────────────────────────────────────────────

# os.makedirs(FAISS_DIR, exist_ok=True)

# # Mongo setup
# client  = MongoClient(MONGO_URI)
# db      = client[DB_NAME]
# fac_c   = db[FAC_COLL]
# dept_c  = db[DEPT_COLL]

# # embedding model
# embedder = SentenceTransformer(EMBED_MODEL_NAME)


# def rebuild_sqlite_faiss_excel():
#     # 1) aggregate with department lookup
#     pipeline = [
#         {"$lookup": {
#             "from": DEPT_COLL,
#             "localField": "dept_id",
#             "foreignField": "_id",
#             "as": "dept_docs"
#         }},
#         {"$unwind": {"path": "$dept_docs", "preserveNullAndEmptyArrays": True}},
#         {"$project": {
#             "_id":            1,
#             "salutation":     1,
#             "firstname":      1,
#             "surname":        1,
#             "email":          1,
#             "photo":          1,
#             "overview":       1,
#             "designation":    1,
#             "faculty_type":   1,
#             "research":       1,
#             "department_name":"$dept_docs.name"
#         }}
#     ]
#     docs = list(fac_c.aggregate(pipeline))

#     # 2) flatten into DataFrame
#     rows = []
#     for d in docs:
#         full_name = " ".join(filter(None, [
#             d.get("salutation", "").strip(),
#             d.get("firstname", "").strip(),
#             d.get("surname", "").strip()
#         ]))
#         rows.append({
#             "_id":            str(d["_id"]),
#             "full_name":      full_name,
#             "email":          d.get("email", ""),
#             "designation":    d.get("designation", ""),
#             "department":     d.get("department_name", ""),
#             "faculty_type":   d.get("faculty_type", ""),
#             "overview":       d.get("overview", ""),
#             "photo":          d.get("photo", ""),
#             "research":       ";".join(str(r) for r in d.get("research", []))
#         })
#     if not rows:
#         print(f"[{time.ctime()}] ⚠️  No faculty records found, skipping SQLite/Excel/FAISS rebuild.")
#         return

#     df = pd.DataFrame(rows)

#     # 3) Write to SQLite
    
#     # 3) Write to SQLite
#     conn = sqlite3.connect(SQLITE_PATH)
#     df.to_sql("faculties", conn, if_exists="replace", index=False)
#     conn.close()
#     print(f"[{time.ctime()}] Wrote {len(df)} records to SQLite ({SQLITE_PATH})")
#     # 4) Write to Excel
#     df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
#     print(f"[{time.ctime()}] Wrote {len(df)} records to Excel ({EXCEL_PATH})")

#     # 5) Build FAISS on full_name embeddings
#     names = df["full_name"].fillna("").tolist()
#     embs  = embedder.encode(names, show_progress_bar=False)
#     xb    = np.array(embs, dtype="float32")

#     dim = xb.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(xb)

#     # 6a) Save native FAISS
#     faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
#     # 6b) Save pickle
#     with open(os.path.join(FAISS_DIR, "index.pkl"), "wb") as f:
#         pickle.dump(index, f)

#     print(f"[{time.ctime()}] Saved FAISS index ({len(xb)} vectors) → {FAISS_DIR}/")


# def watch_and_rebuild():
#     # Initial build
#     rebuild_sqlite_faiss_excel()

#     # Watch for changes
#     with fac_c.watch() as stream:
#         print(f"[{time.ctime()}] Watching `{FAC_COLL}` for changes…")
#         for change in stream:
#             print(f"[{time.ctime()}] Change detected: {change['operationType']}")
#             rebuild_sqlite_faiss_excel()


# if __name__ == "__main__":
#     watch_and_rebuild()




#!/usr/bin/env python3
import os
import time
import sqlite3

import pandas as pd
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
MONGO_URI        = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
DB_NAME          = "dseu-data"
FAC_COLL         = "faculties"
DEPT_COLL        = "departments"

SQLITE_PATH      = "faculties.db"
EXCEL_PATH       = "faculties.xlsx"
FAISS_DIR        = "data/faculties_folder"  # will contain both index files and metadata
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(FAISS_DIR, exist_ok=True)

# MongoDB setup
client  = MongoClient(MONGO_URI)
db      = client[DB_NAME]
fac_c   = db[FAC_COLL]
dept_c  = db[DEPT_COLL]

# LangChain embedding wrapper (for FAISS)
lc_embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def rebuild_sqlite_faiss_excel():
    # 1) Join faculties with department documents
    pipeline = [
        {"$lookup": {
            "from": DEPT_COLL,
            "localField": "dept_id",
            "foreignField": "_id",
            "as": "dept_docs"
        }},
        {"$unwind": {"path": "$dept_docs", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id":            1,
            "salutation":     1,
            "firstname":      1,
            "surname":        1,
            "email":          1,
            "photo":          1,
            "overview":       1,
            "designation":    1,
            "faculty_type":   1,
            "research":       1,
            "department_name":"$dept_docs.name"
        }}
    ]
    docs = list(fac_c.aggregate(pipeline))

    # 2) Flatten to DataFrame
    rows = []
    for d in docs:
        full_name = " ".join(filter(None, [
            d.get("salutation", "").strip(),
            d.get("firstname", "").strip(),
            d.get("surname", "").strip()
        ]))
        rows.append({
            "_id":          str(d["_id"]),
            "full_name":    full_name,
            "email":        d.get("email", ""),
            "designation":  d.get("designation", ""),
            "department":   d.get("department_name", ""),
            "faculty_type": d.get("faculty_type", ""),
            "overview":     d.get("overview", ""),
            "photo":        d.get("photo", ""),
            "research":     ";".join(map(str, d.get("research", [])))
        })
    if not rows:
        print(f"[{time.ctime()}] ⚠️ No faculty records found; skipping rebuild.")
        return

    df = pd.DataFrame(rows)

    # 3) Write to SQLite
    with sqlite3.connect(SQLITE_PATH) as conn:
        df.to_sql("faculties", conn, if_exists="replace", index=False)
    print(f"[{time.ctime()}] Wrote {len(df)} records to SQLite ({SQLITE_PATH})")

    # 4) Write to Excel
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
    print(f"[{time.ctime()}] Wrote {len(df)} records to Excel ({EXCEL_PATH})")

    # 5) Build & save FAISS vectorstore
    names = df["full_name"].fillna("").tolist()
    vectorstore = FAISS.from_texts(
        texts=names,
        embedding=lc_embedding
    )
    vectorstore.save_local(FAISS_DIR)
    print(f"[{time.ctime()}] Saved FAISS vectorstore with {len(names)} entries → {FAISS_DIR}/")


def watch_and_rebuild():
    # Initial build
    rebuild_sqlite_faiss_excel()
    print(f"[{time.ctime()}] Watching `{FAC_COLL}` for changes…")
    with fac_c.watch() as stream:
        for change in stream:
            print(f"[{time.ctime()}] Change detected: {change['operationType']}")
            rebuild_sqlite_faiss_excel()


if __name__ == "__main__":
    watch_and_rebuild()
