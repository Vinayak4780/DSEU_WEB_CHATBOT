#!/usr/bin/env python3
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task
from shared_llm import shared_llm as llm

# — FACULTY PROMPT WITH CHAT HISTORY & CONTEXT —
faculty_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
{chat_history}

Use only the following flattened fields in the retrieved context:

{context}

Question: {question}
"""
)

# — EMBEDDING MODEL & VECTORSTORE —
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/faculties_folder",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(k=3)

# — SHARED CONVERSATION MEMORY —
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=False
)

# — BUILD THE FACULTY QA CHAIN —
faculty_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": faculty_prompt,
        "document_variable_name": "context"
    },
    return_source_documents=False
)

def faculty_bot(user_input: str) -> str:
    """Run the faculty QA chain on each user query and return the answer."""
    return faculty_qa.run(user_input)

# — CREWAI AGENT & TASK FOR FACULTY —
faculty_agent = Agent(
    role="DSEU Faculty Bot",
    goal=(
        "Answer queries about DSEU faculty members and their profiles, "
        "using only the defined faculty fields."
    ),
    backstory=(
        "You have access to these flattened faculty fields:\n"
        "- full_name, email, designation, department, faculty_type,\n"
        "- overview, photo, research\n\n"
        "Use only these fields to provide accurate faculty information."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

faculty_task = Task(
    description="Answer questions about DSEU faculty members and their profiles.",
    expected_output="Fact-based answers referencing only the permitted faculty fields.",
    agent=faculty_agent,
    async_execution=False,
    callback=lambda x: faculty_bot(x.input)
)
