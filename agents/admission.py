from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task

# — GENERIC ADMISSION PROMPT —
admission_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
{chat_history}

You are an expert AI assistant on DSEU admissions. Use only the retrieved context to answer.

Question: {question}
"""
)

# — EMBEDDING MODEL —
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# — LOAD TWO FAISS INDEXES —
docs_index = FAISS.load_local(
    "data/admissions/faiss_docs",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
links_index = FAISS.load_local(
    "data/admissions/faiss_meta",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# — RETRIEVERS —
docs_retriever  = docs_index.as_retriever(k=3)
links_retriever = links_index.as_retriever(k=3)

# — SHARED MEMORY & LLM —
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
from shared_llm import shared_llm as llm

# — QA CHAINS —
combine_kwargs = {
    "prompt": admission_prompt,
    "document_variable_name": "chat_history",
}
docs_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docs_retriever,
    memory=memory,
    combine_docs_chain_kwargs=combine_kwargs,
    return_source_documents=False,
)
links_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=links_retriever,
    memory=memory,
    combine_docs_chain_kwargs=combine_kwargs,
    return_source_documents=False,
)


# — ROUTING HELPERS —
def is_admission_query(q: str) -> bool:
    return "admission" in q.lower()

def is_link_query(q: str) -> bool:
    return any(term in q.lower() for term in ["link", "pdf", "download", "url"])

def main_bot(user_input: str) -> str:
    # only handle admission-related questions
    if not is_admission_query(user_input):
        return "Sorry, I only handle DSEU admission queries right now."
    # if they ask for a link/download → use link index
    if is_link_query(user_input):
        return links_qa.run(user_input)
    # otherwise → return passage from the docs index
    return docs_qa.run(user_input)

# — CREWAI AGENT & TASK —
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
    callback=lambda x: main_bot(x.input)
)




##!/usr/bin/env python3
# """
# admission_bot.py

# Loads your pre-built FAISS indices (meta & docs) via the loader functions in
# notices_indexer.py, then spins up a simple REPL for DSEU admission Q&A.
# """

# import logging

# # 1) Correct embedding import
# from langchain_huggingface import HuggingFaceEmbeddings

# # 2) Your original loader functions
# from langchain_community.vectorstores import FAISS

# # 3) LangChain & CrewAI imports
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from crewai import Agent, Task
# from shared_llm import shared_llm as llm


# # ─── CONFIG & EMBEDDING MODEL ──────────────────────────────────────────────────
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# META_FAISS_DIR  = "data/admissions/faiss_meta"
# DOC_FAISS_DIR   = "data/admissions/faiss_docs"
# DOC_TEXT_DIR    = "data/admissions/faiss_docs_texts"

# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# # ─── LOAD INDEXES (via your own functions) ────────────────────────────────────
# docs_index = FAISS.load_local(
#      DOC_FAISS_DIR,
#      embeddings,
#      allow_dangerous_deserialization=True,
# )
# links_index = FAISS.load_local(
#      META_FAISS_DIR,
#      embeddings,
#      allow_dangerous_deserialization=True,)

# docs_retriever  = docs_index.as_retriever(search_kwargs={"k": 1})
# links_retriever = links_index.as_retriever(search_kwargs={"k": 1})


# # ─── PROMPT (context + question) ───────────────────────────────────────────────
# admission_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# Relevant context:
# {context}

# You are an expert AI assistant on DSEU admissions. Use only the retrieved context to answer.

# Question: {question}
# """
# )


# # ─── BUILD YOUR CHAINS ─────────────────────────────────────────────────────────
# docs_qa = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=docs_retriever,
#     memory=None,
#     combine_docs_chain_kwargs={"prompt": admission_prompt},
#     return_source_documents=False,
# )
# links_qa = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=links_retriever,
#     memory=None,
#     combine_docs_chain_kwargs={"prompt": admission_prompt},
#     return_source_documents=False,
# )


# # ─── ROUTING & BOT LOGIC ───────────────────────────────────────────────────────
# def is_admission_query(q: str) -> bool:
#     return "admission" in q.lower()

# def is_link_query(q: str) -> bool:
#     return any(t in q.lower() for t in ("link", "pdf", "download", "url"))

# def main_bot(user_input: str) -> str:
#     if not is_admission_query(user_input):
#         return "Sorry, I only handle DSEU admission queries right now."
#     chain = links_qa if is_link_query(user_input) else docs_qa
#     # .run(text) will automatically bind to {"question": text} under the hood
#     return chain.run(user_input)


# # ─── REPL ENTRYPOINT ──────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO,
#                         format="%(asctime)s [%(levelname)s] %(message)s")

#     print("DSEU Admission Bot ready. Ask your question:")
#     while True:
#         q = input("You: ")
#         print("Bot:", main_bot(q))


# # ─── Optional CrewAI Agent Setup ───────────────────────────────────────────────
# admission_agent = Agent(
#     role="DSEU Admission Bot",
#     goal="Answer DSEU admission questions, returning full-text answers or PDF links.",
#     backstory="Two FAISS indexes: docs & meta.",
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )

# admission_task = Task(
#     description=(
#         "Answer questions about DSEU admission cycles, deadlines, "
#         "eligibility, and application details."
#     ),
#     expected_output="Accurate answers or download URLs.",
#     agent=admission_agent,
#     async_execution=False,
#     callback=main_bot,
# )
