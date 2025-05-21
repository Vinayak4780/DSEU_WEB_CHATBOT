# 1. Install dependencies if you haven’t already:
#    pip install langchain langchain-community crewai sentence-transformers faiss-cpu

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from crewai import Agent, Task
from shared_llm import shared_llm as llm
# 6. Build the conversational retrieval chain
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/defaultllm_docs",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Set up conversational memory (now telling it to save only "answer")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# 4. Define the prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are a knowledgeable assistant. Leverage the conversation history and retrieved context 
to answer the user’s question as accurately as possible.

Conversation History:
{chat_history}

Retrieved Context:
{context}

Question: {question}
"""
)

# 6. Build the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 7. Wrap it as a simple function
def document_qa(query: str) -> str:
    result = qa_chain({"question": query})
    return result["answer"]

# 8. Create the CrewAI Agent
default_agent = Agent(
    role="DSEU Admission Bot",
    goal=(
        "Answer the user’s question by running the document QA chain over the vector store."
    ),
    backstory=(
        "You are a knowledgeable assistant. Leverage the conversation history and retrieved context "
        "to answer the user’s question as accurately as possible."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)
default_task = Task(
    description="Answer the user’s question by running the document QA chain over the vector store.",
    expected_output="A natural-language answer to the user’s query",
    agent=default_agent,
    async_execution=False,
    callback=lambda x: document_qa(x.input)
)

# 9. Interact loop
if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        print("Agent:", document_qa(user_input))
