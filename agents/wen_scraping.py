import os
import time

# LangChain + FAISS + Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# CrewAI agent/task
from crewai import Agent, Task

# Shared LLM instance
from shared_llm import shared_llm as llm

# ─── WEB-SCRAPING QA SETUP ────────────────────────────────────────────────────

web_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
{chat_history}

You are an expert AI assistant on web-scraped content.

Use **only** the following retrieved context to answer:

{context}

Question: {question}
"""
)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "data/web_scraping"

embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
web_index  = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

web_retriever = web_index.as_retriever(k=3)
memory        = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

web_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=web_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": web_prompt},
    return_source_documents=False,
)

def web_bot(user_input: str) -> str:
    """Route user question into the conversational chain."""
    return web_qa.run(user_input)


# 6) CrewAI Agent & Task for orchestrating this QA bot
web_agent = Agent(
    role="Web-Scrape QA Bot",
    goal=(
        "Answer questions about the web-scraped dataset, "
        "using retrieved document snippets in a conversational manner."
    ),
    backstory=(
        "You have a FAISS index over web-scraped content at `data/web_scraping`.\n"
        "Use the retrieval chain with chat history to form context-aware replies."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

web_task = Task(
    description="Answer queries on the web-scraped FAISS index via a conversational chain.",
    expected_output="Context-aware answers drawn only from the retrieved web-scraped content.",
    agent=web_agent,
    async_execution=False,
    callback=web_bot
)
