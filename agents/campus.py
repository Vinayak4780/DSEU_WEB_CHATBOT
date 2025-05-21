#!/usr/bin/env python3
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task
from shared_llm import shared_llm as llm

# — UPDATED PROMPT FOR PROGRAM DATA —
pipeline_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
{chat_history}

You are an expert AI assistant helping users query DSEU campus information.

Use **only** the following flattened fields (they are provided as `context`):
{context}

Answer the user’s question using only these fields.

Question: {question}
"""
)

# — EMBEDDING MODEL & VECTORSTORE —
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/campus_faiss",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(k=3)

# — MEMORY & LLM —
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=False
)

# — BUILD THE QA CHAIN —
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": pipeline_prompt,
        "document_variable_name": "context"
    },
    return_source_documents=False
)

def syllabus_bot(user_input: str) -> str:
    """Run the program QA chain and return an answer."""
    return qa_chain.run(user_input)

# — CREWAI AGENT & TASK FOR PROGRAM —
campus_agent = Agent(
    role="DSEU Program Bot",
    goal=(
        "Answer student queries about DSEU program structure, credits, exit options, "
        "and syllabi using only the defined program fields."
    ),
    backstory=(
        "You have access to these flattened program fields:\n"
        "- campus_director, campus_director_email, campus_message, location, campus_info,\n"
        "- courses_offered, labs, zone, program_id, name, programLevel, duration, mode, exit_options, program_department,\n"
        "Use only these fields to provide precise, context-aware answers."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

campus_task = Task(
    description="Answer questions about DSEU program details: credits, exit options, and syllabi.",
    expected_output="Fact-based answers referencing only the permitted program fields.",
    agent=campus_agent,
    async_execution=False,
    callback=lambda x: syllabus_bot(x.input)
)
