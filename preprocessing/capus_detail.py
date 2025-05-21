

# import threading
# import time

# from pymongo import MongoClient
# import pandas as pd

# def fetch_and_normalize(db):
#     pipeline = [
#         # 1️⃣ Convert your string IDs in program_id → ObjectId
#         {
#             "$addFields": {
#                 "program_obj_ids": {
#                     "$map": {
#                         "input": "$program_id",
#                         "as":    "pid",
#                         "in":    { "$toObjectId": "$$pid" }
#                     }
#                 }
#             }
#         },
#         # 2️⃣ Lookup into programs using those new ObjectIds
#         {
#             "$lookup": {
#                 "from":         "programs",
#                 "localField":   "program_obj_ids",
#                 "foreignField": "_id",
#                 "as":           "programs_info"
#             }
#         },
#         # 3️⃣ Project only what you need from campus + reshape programs_info
#         {
#             "$project": {
#                 "_id": 0,
#                 # campus fields
#                 "campus_director":       1,
#                 "campus_director_email": 1,
#                 "campus_message":        1,
#                 "location":              1,
#                 "campus_info":           1,
#                 "courses_offered":       1,
#                 "labs":                  1,
#                 "zone":                  1,

#                 # flatten each program sub‐document
#                 "programs_info": {
#                     "$map": {
#                         "input": "$programs_info",
#                         "as":    "p",
#                         "in": {
#                             "program_id":   "$$p._id",
#                             "name":         "$$p.name",
#                             "programLevel": "$$p.programLevel",
#                             "duration":     "$$p.duration",
#                             "mode":         "$$p.mode",
#                             "exit_options": "$$p.exit_options",
#                             "years":        "$$p.years",
#                             "dept_id":      "$$p.department"
#                         }
#                     }
#                 }
#             }
#         }
#     ]

#     # run aggregation
#     docs = list(db.campus.aggregate(pipeline))

#     # explode programs_info → one row per (campus × program)
#     df = pd.json_normalize(
#         docs,
#         record_path=['programs_info'],
#         meta=[
#             'campus_director',
#             'campus_director_email',
#             'campus_message',
#             'location',
#             'campus_info',
#             'courses_offered',
#             'labs',
#             'zone'
#         ],
#         errors='ignore'
#     )

#     # ─── flatten years ───────────────────────────────────────────────────
#     if 'years' in df.columns:
#         years = pd.json_normalize(df['years'])
#         years.columns = [c.replace('.', '_') for c in years.columns]
#         df = pd.concat([df.drop(columns=['years']), years], axis=1)

#     # ─── flatten exit_options ────────────────────────────────────────────
#     df['exit_options'] = df.get('exit_options', []).apply(
#         lambda v: ', '.join(v) if isinstance(v, list) else (v or '')
#     )

#     # ─── map dept_id → dept_name ─────────────────────────────────────────
#     dept_docs = list(db.departments.find({}, {'_id':1, 'name':1}))
#     dept_map  = {d['_id']: d['name'] for d in dept_docs}
#     df['program_department'] = df.get('dept_id').map(dept_map).fillna('')
#     df.drop(columns=['dept_id'], inplace=True)

#     return df

# def export_to_excel(df, path):
#     df.to_excel(path, index=False)
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Exported {len(df)} rows to {path}")

# def watch_and_export(db, output_path):
#     # watch for insert/update/replace on campus
#     pipeline = [{"$match": {"operationType": {"$in": ["insert","update","replace"]}}}]
#     with db.campus.watch(pipeline, full_document='updateLookup') as stream:
#         for change in stream:
#             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Change detected: {change['operationType']}")
#             df = fetch_and_normalize(db)
#             export_to_excel(df, output_path)

# def main():
#     MONGO_URI   = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
#     DB_NAME     = "dseu-data"
#     OUTPUT_FILE = "campus_program_export.xlsx"

#     client = MongoClient(MONGO_URI)
#     db     = client[DB_NAME]

#     # initial dump
#     df = fetch_and_normalize(db)
#     export_to_excel(df, OUTPUT_FILE)

#     # start background watcher
#     watcher = threading.Thread(
#         target=watch_and_export,
#         args=(db, OUTPUT_FILE),
#         daemon=True
#     )
#     watcher.start()

#     print("Watching for changes… (CTRL+C to quit)")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("\nShutting down.")

# if __name__ == "__main__":
#     main()






























# import os
# import threading
# import time
# import pickle
# import sqlite3

# from pymongo import MongoClient
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss

# # ─── CONFIG ─────────────────────────────────────────────────────────
# MONGO_URI        = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
# DB_NAME          = "dseu-data"
# OUTPUT_EXCEL     = "campus_program_export.xlsx"

# # where to store FAISS index, metadata, and SQLite DB
# FAISS_DIR        = "campus_faiss"
# os.makedirs(FAISS_DIR, exist_ok=True)

# # embedding model
# EMB_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
# model            = SentenceTransformer(EMB_MODEL_NAME)


# # ─── DATA NORMALIZATION ─────────────────────────────────────────────
# def fetch_and_normalize(db):
#     pipeline = [
#         {
#             "$addFields": {
#                 "program_obj_ids": {
#                     "$map": {
#                         "input": "$program_id",
#                         "as":    "pid",
#                         "in":    { "$toObjectId": "$$pid" }
#                     }
#                 }
#             }
#         },
#         {
#             "$lookup": {
#                 "from":         "programs",
#                 "localField":   "program_obj_ids",
#                 "foreignField": "_id",
#                 "as":           "programs_info"
#             }
#         },
#         {
#             "$project": {
#                 "_id": 0,
#                 "campus_director":       1,
#                 "campus_director_email": 1,
#                 "campus_message":        1,
#                 "location":              1,
#                 "campus_info":           1,
#                 "courses_offered":       1,
#                 "labs":                  1,
#                 "zone":                  1,

#                 "programs_info": {
#                     "$map": {
#                         "input": "$$ROOT.programs_info",
#                         "as":    "p",
#                         "in": {
#                             "program_id":   "$$p._id",
#                             "name":         "$$p.name",
#                             "programLevel": "$$p.programLevel",
#                             "duration":     "$$p.duration",
#                             "mode":         "$$p.mode",
#                             "exit_options": "$$p.exit_options",
#                             "years":        "$$p.years",
#                             "dept_id":      "$$p.department"
#                         }
#                     }
#                 }
#             }
#         }
#     ]

#     docs = list(db.campus.aggregate(pipeline))
#     df = pd.json_normalize(
#         docs,
#         record_path=['programs_info'],
#         meta=[
#             'campus_director',
#             'campus_director_email',
#             'campus_message',
#             'location',
#             'campus_info',
#             'courses_offered',
#             'labs',
#             'zone'
#         ],
#         errors='ignore'
#     )

#     # flatten years
#     if 'years' in df.columns:
#         yrs = pd.json_normalize(df.pop('years'))
#         yrs.columns = [c.replace('.', '_') for c in yrs.columns]
#         df = pd.concat([df, yrs], axis=1)

#     # flatten exit_options, courses_offered & labs lists → comma-joined
#     for col in ['exit_options', 'courses_offered', 'labs']:
#         if col in df.columns:
#             df[col] = df[col].apply(
#                 lambda v: ', '.join(v) if isinstance(v, list) else (v or '')
#             )

#     # convert any ObjectId or other non-primitive to string for sqlite
#     df = df.applymap(lambda x: x if isinstance(x, (str, int, float, type(None))) else str(x))

#     # map dept_id → dept_name
#     dept_docs = list(db.departments.find({}, {'_id':1, 'name':1}))
#     dept_map  = {d['_id']: d['name'] for d in dept_docs}
#     if 'dept_id' in df.columns:
#         df['program_department'] = df.pop('dept_id').map(dept_map).fillna('')

#     return df


# # ─── EXPORT FUNCTIONS ────────────────────────────────────────────────
# def export_to_excel(df, path):
#     df.to_excel(path, index=False)
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Exported {len(df)} rows to {path}")


# def export_to_sqlite(df, db_path):
#     # ensure pure sqlite-friendly types
#     df_clean = df.copy()
#     df_clean = df_clean.applymap(lambda x: x if isinstance(x, (str, int, float, type(None))) else str(x))
#     conn = sqlite3.connect(db_path)
#     df_clean.to_sql("campus_program", conn, if_exists="replace", index=False)
#     conn.close()
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Wrote {len(df_clean)} rows to SQLite DB {db_path}")


# def build_and_save_faiss(df):
#     texts = (
#         df.get("campus_message", pd.Series()).fillna("") + " | " +
#         df.get("name", pd.Series()).fillna("")           + " | " +
#         df.get("program_department", pd.Series()).fillna("")
#     ).tolist()

#     embs = model.encode(texts, show_progress_bar=True).astype("float32")
#     dim  = embs.shape[1]

#     index = faiss.IndexFlatL2(dim)
#     index.add(embs)

#     faiss_path = os.path.join(FAISS_DIR, "index.faiss")
#     meta_path  = os.path.join(FAISS_DIR, "index.pkl")

#     faiss.write_index(index, faiss_path)
#     with open(meta_path, "wb") as f:
#         pickle.dump(df.to_dict(orient="records"), f)

#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAISS index → {faiss_path}")
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Metadata pickle → {meta_path}")


# # ─── WATCHER ─────────────────────────────────────────────────────────
# def watch_and_export(db, output_excel):
#     pipeline = [{"$match": {"operationType": {"$in": ["insert","update","replace"]}}}]
#     with db.campus.watch(pipeline, full_document='updateLookup') as stream:
#         for change in stream:
#             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Change detected: {change['operationType']}")
#             df = fetch_and_normalize(db)
#             export_to_excel(df, output_excel)
#             build_and_save_faiss(df)
#             export_to_sqlite(df, os.path.join(FAISS_DIR, "campus_program.db"))


# # ─── MAIN ────────────────────────────────────────────────────────────
# def main():
#     client = MongoClient(MONGO_URI)
#     db     = client[DB_NAME]

#     # initial dump & FAISS & SQLite export
#     df = fetch_and_normalize(db)
#     export_to_excel(df, OUTPUT_EXCEL)
#     build_and_save_faiss(df)
#     export_to_sqlite(df, os.path.join(FAISS_DIR, "campus_program.db"))

#     # start watcher thread
#     watcher = threading.Thread(
#         target=watch_and_export,
#         args=(db, OUTPUT_EXCEL),
#         daemon=True
#     )
#     watcher.start()

#     print("Watching for changes… (CTRL+C to quit)")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("\nShutting down.")


# if __name__ == "__main__":
#     main()

































import os
import threading
import time
import pickle
import sqlite3
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# ─── CONFIG ─────────────────────────────────────────────────────────
MONGO_URI        = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
DB_NAME          = "dseu-data"
OUTPUT_EXCEL     = "campus_program_export.xlsx"

# where to store FAISS index, metadata, and SQLite DB
FAISS_DIR        = "data/campus_faiss"
os.makedirs(FAISS_DIR, exist_ok=True)

# embedding model
EMB_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
model            = SentenceTransformer(EMB_MODEL_NAME)

# ─── DATA NORMALIZATION ─────────────────────────────────────────────
def fetch_and_normalize(db):
    # Generic fetch: join campus->programs->departments
    pipeline = [
        {"$addFields": {
            "program_obj_ids": {"$map": {"input": "$program_id", "as": "pid", "in": {"$toObjectId": "$$pid"}}}
        }},
        {"$lookup": {"from": "programs", "localField": "program_obj_ids", "foreignField": "_id", "as": "programs_info"}},
        {"$unwind": "$programs_info"},
        {"$project": {
            "_id": 0,
            # campus fields
            "campus_director":       1,
            "campus_director_email": 1,
            "campus_message":        1,
            "location":              1,
            "campus_info":           1,
            "courses_offered":       1,
            "labs":                  1,
            "zone":                  1,
            # program fields
            "program_id":            "$programs_info._id",
            "name":                  "$programs_info.name",
            "programLevel":          "$programs_info.programLevel",
            "duration":              "$programs_info.duration",
            "mode":                  "$programs_info.mode",
            "exit_options":          "$programs_info.exit_options",
            "years":                 "$programs_info.years",
            "dept_id":               "$programs_info.department"
        }}
    ]
    docs = list(db.campus.aggregate(pipeline))
    df = pd.json_normalize(docs)

    # flatten dynamic lists and dicts
    list_cols = ['exit_options','courses_offered','labs']
    for col in list_cols:
        if col in df:
            df[col] = df[col].apply(lambda v: ', '.join(v) if isinstance(v, list) else (v or ''))

    # flatten years dict into columns if present
    if 'years' in df:
        years_df = pd.json_normalize(df.pop('years')).add_prefix('year_')
        df = pd.concat([df, years_df], axis=1)

    # stringify non-primitives
    df = df.applymap(lambda x: x if isinstance(x, (str,int,float,type(None))) else str(x))

    # map dept_id → department name
    dept_map = {d['_id']:d['name'] for d in db.departments.find({},{'_id':1,'name':1})}
    df['program_department'] = df['dept_id'].map(dept_map).fillna('') if 'dept_id' in df else ''
    df.drop(columns=[c for c in ['dept_id','program_obj_ids','years'] if c in df], inplace=True)

    return df

# ─── EXPORT ──────────────────────────────────────────────────────────
def export_all(df):
    # … your Excel + SQLite exports …

    # 1) Build your FAISS index
    texts = (df["campus_message"] + " | " + df["name"] + " | " + df["program_department"]).tolist()
    embs  = model.encode(texts, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    # 2) Build the in-memory docstore + ID map
    records = df.to_dict(orient="records")
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, rec in enumerate(records):
        doc_id = str(i)
        docstore.add({doc_id: Document(page_content=texts[i], metadata=rec)})
        index_to_docstore_id[i] = doc_id

    # 3) Wrap in FAISS and save
    vs = FAISS(model, index, docstore, index_to_docstore_id)
    vs.save_local(FAISS_DIR)    # writes index.faiss + index.pkl correctly

    print(f"FAISS index and pickle saved to {FAISS_DIR}")

# ─── WATCHER ─────────────────────────────────────────────────────────
def watch_and_rebuild(db):
    # watch campus, programs, departments for any change
    pipeline = [{
        '$match': {
            'ns.db': DB_NAME,
            'ns.coll': {'$in': ['campus','programs','departments']},
            'operationType': {'$in': ['insert','update','replace','delete']}
        }
    }]
    with db.watch(pipeline, full_document='updateLookup') as stream:
        print(f"Watching collections: campus, programs, departments...")
        for change in stream:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Change: {change['ns']['coll']} {change['operationType']}")
            df = fetch_and_normalize(db)
            export_all(df)

# ─── MAIN ───────────────────────────────────────────────────────────
def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    # initial build
    df = fetch_and_normalize(db)
    export_all(df)
    # start watcher
    watch_and_rebuild(db)

if __name__=='__main__':
    main()
