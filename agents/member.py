from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task

# — BOARD MEMBERS PROMPT WITH CHAT HISTORY —
board_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
{chat_history}

Use the following context to answer the question:

{context}

Question: {question}
"""
)


# — EMBEDDING MODEL —
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# — LOAD TWO FAISS INDEXES —
#   1) Full-text board-member documents
docs_index = FAISS.load_local(
    "data/members/faiss_docs",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
#   2) Per-member document link pointers
links_index = FAISS.load_local(
    "data/members/faiss_meta",
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
    combine_docs_chain_kwargs={"prompt": board_prompt},
    return_source_documents=False
)
links_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=links_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": board_prompt},
    return_source_documents=False
)
# — ROUTING HELPER —
def is_link_query(text: str) -> bool:
    text = text.lower()
    return any(term in text for term in ["link", "url", "download", "pdf"])

def board_bot(user_input: str) -> str:
    """Route to the appropriate QA chain based on whether links were requested."""
    if is_link_query(user_input):
        return links_qa.run(user_input)
    return docs_qa.run(user_input)

# — CREWAI AGENT & TASK FOR BOARD MEMBERS —
board_agent = Agent(
    role="DSEU Board Bot",
    goal=(
        "Answer queries about DSEU board members (e.g., Vice Chancellor, Registrar, Deans), "
        "returning either full-text context or document links when explicitly requested."
    ),
    backstory=(
        "You have two FAISS indexes:\n"
        "1) Over full-text board-member documents (data/board_faiss)\n"
        "2) Over link pointers to each board-member document (data/board_links_faiss)\n\n"
        "If the user asks for a link or download, use the link index; otherwise use the text index."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

board_task = Task(
    description="Answer questions about DSEU board members and their profiles, providing links on request.",
    expected_output="Fact-based answers using retrieved document text or PDF links as appropriate.",
    agent=board_agent,
    async_execution=False,
    callback=lambda x: board_bot(x.input)
)
