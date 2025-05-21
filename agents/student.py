from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task

# — STUDENT QUERIES PROMPT WITH CHAT HISTORY —
student_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
{chat_history}

You are an expert AI assistant helping users query DSEU student information.

Use **only** the following retrieved context to answer:

{context}

Question: {question}
"""
)


# — EMBEDDING MODEL —
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# — LOAD TWO FAISS INDEXES —
#   1) Full-text student document index
docs_index = FAISS.load_local(
    "data/students/faiss_docs",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
#   2) PDF‐link pointer index for student docs
links_index = FAISS.load_local(
    "data/students/faiss_meta",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# — RETRIEVERS —
docs_retriever  = docs_index.as_retriever(k=3)
links_retriever = links_index.as_retriever(k=3)

# — SHARED MEMORY & LLM —
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
from shared_llm import shared_llm as llm

# — BUILD TWO QA CHAINS —
docs_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docs_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": student_prompt},
    return_source_documents=False,
)

links_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=links_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": student_prompt},
    return_source_documents=False,
)

# — ROUTING HELPER —
def is_link_query(text: str) -> bool:
    text = text.lower()
    return any(term in text for term in ["link", "url", "pdf", "download"])

def student_bot(user_input: str) -> str:
    """Route to the appropriate QA chain based on whether links were requested."""
    if is_link_query(user_input):
        return links_qa.run(user_input)
    return docs_qa.run(user_input)

# — CREWAI AGENT & TASK FOR STUDENT QUERIES —
student_agent = Agent(
    role="DSEU Student Bot",
    goal=(
        "Answer queries about DSEU student information and related documents, "
        "providing either full-text context or download links when explicitly requested."
    ),
    backstory=(
        "You have two FAISS indexes:\n"
        "1) Over full-text student documents (data/students/faiss_docs)\n"
        "2) Over PDF link pointers to each student document (data/students/faiss_links)\n\n"
        "If the user asks for a link or download, use the link index; otherwise use the text index."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

student_task = Task(
    description="Answer questions about DSEU student information, returning text or links on request.",
    expected_output="Accurate, context-aware answers using retrieved student document text or PDF links as appropriate.",
    agent=student_agent,
    async_execution=False,
    callback=lambda x: student_bot(x.input)
)
