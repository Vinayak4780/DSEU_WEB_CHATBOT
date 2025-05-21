#!/usr/bin/env python3
import os
import pickle
import logging

import faiss
from pymongo import MongoClient
from langchain.schema import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MONGO_URI       = "mongodb+srv://osditautomation:8uQPiEsJJyViy6j6@dseu-data.s19nz.mongodb.net/dseu-data?retryWrites=true&w=majority"
DB_NAME         = "dseu-data"
COLLECTION_NAME = "notices"
SECTION_NAME    = "admission"

# ─── EMBEDDINGS ────────────────────────────────────────────────────────────────
EMB = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ─── SIMPLE DOCSTORE WRAPPER ───────────────────────────────────────────────────
class SimpleDocstore:
    def __init__(self, mapping: dict[str, Document]):
        self._mapping = mapping
    def search(self, key: str) -> Document:
        return self._mapping[key]

# ─── FETCHER ──────────────────────────────────────────────────────────────────
def fetch_notices():
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][COLLECTION_NAME].find({"section": SECTION_NAME}))
    client.close()
    return docs

# ─── LOADER FOR DOCS INDEX (full PDF text) ────────────────────────────────────
def load_doc_vectorstore(faiss_folder: str, texts_folder: str):
    index = faiss.read_index(os.path.join(faiss_folder, "index.faiss"))
    with open(os.path.join(faiss_folder, "index.pkl"), "rb") as f:
        ids = pickle.load(f)
    docstore_dict = {}
    for _id in ids:
        path = os.path.join(texts_folder, f"{_id}.txt")
        content = open(path, encoding="utf-8").read()
        docstore_dict[_id] = Document(page_content=content, metadata={"source": path})
    index_to_docstore_id = {i: _id for i, _id in enumerate(ids)}
    return FAISS(EMB, index, SimpleDocstore(docstore_dict), index_to_docstore_id)

# ─── LOADER FOR META INDEX (filename + link) ──────────────────────────────────
def load_meta_vectorstore(faiss_folder: str):
    index = faiss.read_index(os.path.join(faiss_folder, "index.faiss"))
    with open(os.path.join(faiss_folder, "index.pkl"), "rb") as f:
        ids = pickle.load(f)
    raw = fetch_notices()
    docstore_dict = {}
    for d in raw:
        key = str(d["_id"])
        text = f"{d.get('fileName','')} {d.get('fileLink','')}"
        docstore_dict[key] = Document(
            page_content=text,
            metadata={"fileName": d.get("fileName"), "fileLink": d.get("fileLink")}            
        )
    index_to_docstore_id = {i: k for i, k in enumerate(ids)}
    return FAISS(EMB, index, SimpleDocstore(docstore_dict), index_to_docstore_id)

# ─── BUILD VECTORSTORES ───────────────────────────────────────────────────────
docs_index  = load_doc_vectorstore("data/admissions/faiss_docs", "admissions/faiss_docs_texts")
links_index = load_meta_vectorstore("data/admissions/faiss_meta")

# ─── RETRIEVERS (top-1) ───────────────────────────────────────────────────────
docs_retriever  = docs_index.as_retriever(search_kwargs={"k": 1})
links_retriever = links_index.as_retriever(search_kwargs={"k": 1})

# ─── PROMPT ────────────────────────────────────────────────────────────────────
admission_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Relevant context:
{context}

You are an expert AI assistant on DSEU admissions. Use only the retrieved context to answer.

Question: {question}
"""
)

from shared_llm import shared_llm as llm

# ─── QA CHAINS ────────────────────────────────────────────────────────────────
docs_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docs_retriever,
    memory=None,
    combine_docs_chain_kwargs={"prompt": admission_prompt},
    return_source_documents=False
)
links_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=links_retriever,
    memory=None,
    combine_docs_chain_kwargs={"prompt": admission_prompt},
    return_source_documents=False
)

# ─── ROUTER ───────────────────────────────────────────────────────────────────
def is_admission_query(q: str) -> bool:
    return "admission" in q.lower()

def is_link_query(q: str) -> bool:
    return any(term in q.lower() for term in ["link", "pdf", "download", "url"])

# Fix main_bot to use .run() which maps positional string to 'question'
def main_bot(user_input: str) -> str:
    if not is_admission_query(user_input):
        return "Sorry, I only handle DSEU admission queries right now."
    inputs = {
        "question": user_input,
        "chat_history": []          # ← supply an empty history
    }
    if is_link_query(user_input):
        out = links_qa.invoke(inputs)
    else:
        out = docs_qa.invoke(inputs)
    return out["answer"]


# ─── CREWAI SETUP ─────────────────────────────────────────────────────────────
admission_agent = Agent(
    role="DSEU Admission Bot",
    goal="Answer DSEU admission questions, returning either full-text answers or PDF links as requested.",
    backstory="You have two FAISS indexes: one over admission document text, one over admission PDF links.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)
admission_task = Task(
    description="Answer questions about DSEU admission cycles, deadlines, eligibility, and application details.",
    expected_output="Accurate, passage-based answers or download URLs for admission documents.",
    agent=admission_agent,
    async_execution=False,
    callback=main_bot
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    while True:
        q = input("You: ")
        resp = main_bot(q)
        print("Bot:", resp)
