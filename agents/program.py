from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task

# — UPDATED PROMPT FOR PROGRAM DATA —
pipeline_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
{chat_history}

You are an expert AI assistant helping users query DSEU program information.

Use **only** the following flattened fields:
- _id  
- name  
- programLevel  
- duration  
- mode  
- department  
- exit_options  
- year1_credit  
- year1_exit  
- year1_syllabus  
- year2_credit  
- year2_exit  
- year2_syllabus  
- year3_credit  
- year3_exit  
- year3_syllabus  
- year4_credit  
- year4_exit  
- year4_syllabus  

Here is the context (the retrieved program records):
{context}

Answer the user’s question using only these fields.

Question: {question}
"""
)


# — EMBEDDING MODEL & VECTORSTORE (unchanged) —
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/program_faiss",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(k=3)

# — MEMORY & LLM (unchanged) —
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
from shared_llm import shared_llm as llm

# — QA CHAIN —
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": pipeline_prompt},
    return_source_documents=False
)

def syllabus_bot(user_input: str) -> str:
    return qa_chain.run(user_input)

# — UPDATED CREWAI AGENT & TASK —
program_agent = Agent(
    role="DSEU Program Bot",
    goal=(
        "Answer student queries about DSEU program structure, credits, exit options, "
        "and syllabi using only the defined program fields."
    ),
    backstory=(
        "You have access to these flattened program fields:\n"
        "- _id, name, programLevel, duration, mode, department,\n"
        "- exit_options,\n"
        "- year1_credit, year1_exit, year1_syllabus,\n"
        "- year2_credit, year2_exit, year2_syllabus,\n"
        "- year3_credit, year3_exit, year3_syllabus,\n"
        "- year4_credit, year4_exit, year4_syllabus\n\n"
        "Use only these fields to provide precise, context-aware answers."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

program_task = Task(
    description="Answer questions about DSEU program details: credits, exit options, and syllabi.",
    expected_output="Fact-based answers referencing only the permitted program fields.",
    agent=program_agent,
    async_execution=False,
    callback=lambda x: syllabus_bot(x.input)
)
