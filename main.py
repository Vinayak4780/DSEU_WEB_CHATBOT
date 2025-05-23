# # # main.py

# # import sys
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity

# # # 1) Import all your individual bot callbacks
# # from agents.admission import main_bot     as admission_bot
# # from agents.program  import syllabus_bot as program_bot
# # from agents.faculties   import faculty_bot  as faculty_bot
# # from agents.member    import board_bot    as board_bot
# # from agents.student   import student_bot  as student_bot
# # from agents.campus import syllabus_bot as campus_bot

# # # 2) Map a short “domain name” → its bot
# # DOMAIN_BOTS = {
# #     "admission": admission_bot,
# #     "program":   program_bot,
# #     "faculty":   faculty_bot,
# #     "board":     board_bot,
# #     "student":   student_bot,
# #     "campus":    campus_bot
# # }

# # # 3) Prepare a TF-IDF router over just these domain names
# # domains = list(DOMAIN_BOTS.keys())
# # vectorizer = TfidfVectorizer().fit(domains)
# # domain_vecs = vectorizer.transform(domains)

# # def route_question(question: str) -> str:
# #     """
# #     1) Vectorize the question
# #     2) Compute cosine similarity against each domain name
# #     3) Pick the best‐matching domain and call its bot
# #     """
# #     q_vec = vectorizer.transform([question])
# #     sims = cosine_similarity(q_vec, domain_vecs)[0]
# #     best_idx = sims.argmax()
# #     best_domain = domains[best_idx]
# #     bot_fn = DOMAIN_BOTS[best_domain]
# #     return bot_fn(question)

# # if __name__ == "__main__":
# #     print("DSEU Unified Assistant (type 'exit' to quit)")
# #     for line in sys.stdin:
# #         user_input = line.strip()
# #         if user_input.lower() in ("exit", "quit"):
# #             break
# #         answer = route_question(user_input)
# #         print(f"\n🤖 {answer}\n")










# # main.py

# # main.py

# import spacy
# from textblob import TextBlob
# from transformers import pipeline
# from crewai import Agent, Task, process
# from langchain.memory import ConversationBufferMemory

# # Import each agent module (make sure your PYTHONPATH includes the project root)
# import agents.admission as admission_agent
# import agents.campus as campus_module
# import agents.default_agent as Default_agent
# import agents.faculties as faculty_agent
# import agents.member as board_agent
# import agents.program as program_agent
# import agents.student as student_agent

# # 1. Preprocessing (lowercase, spelling correction, punctuation removal)
# nlp = spacy.load("en_core_web_sm")

# def preprocess_query(text: str) -> str:
#     corrected = str(TextBlob(text.lower()).correct())
#     doc = nlp(corrected)
#     tokens = [token.text for token in doc if not token.is_punct]
#     return " ".join(tokens)

# # 2. Zero-shot intent classification
# zero_shot = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli",
#     device=-1
# )

# # Build the set of intents from your agent modules
# INTENTS = {
#     "admission": admission_agent,
#     "campus":   campus_module,
#     "faculties":faculty_agent,
#     "member":   board_agent,
#     "program":  program_agent,
#     "student":  student_agent,
# }
# INTENT_LABELS = list(INTENTS.keys()) + ["general_fallback"]
# THRESHOLD = 0.4

# def classify_intent(query: str) -> str:
#     result = zero_shot(query, INTENT_LABELS)
#     label, score = result["labels"][0], result["scores"][0]
#     return label if score >= THRESHOLD else "general_fallback"

# # 3. Setup CrewAI TaskManager with all agents
# all_agents = [
#     admission_agent.agent,
#     campus_module.agent,
#     faculty_agent.agent,
#     board_agent.agent,
#     program_agent.agent,
#     student_agent.agent,
#     Default_agent.agent
# ]
# task_manager = processents=(all_agents)

# # 4. Main handler
# def handle_query(user_query: str) -> str:
#     clean_q = preprocess_query(user_query)
#     intent = classify_intent(clean_q)

#     # select module
#     module = INTENTS.get(intent, Default_agent)

#     # assign task
#     task_manager.assign_task(Task(content=clean_q), module.agent)

#     # try retrieval chain if defined
#     try:
#         if hasattr(module, "chain"):
#             docs = module.chain.retriever.get_relevant_documents(clean_q)
#             if docs:
#                 return module.chain({"question": clean_q})["answer"]
#         # else fallback to agent.run
#         return module.agent.run(clean_q)
#     except Exception:
#         # final fallback
#         return Default_agent.agent.run(clean_q)

# if __name__ == "__main__":
#     test_queries = [
#         "How can I apply for admission?",
#         "Tell me about campus facilities",
#         "Who are the faculty members?",
#         "How do I become a member?",
#         "What programs are offered?",
#         "Show me student clubs",
#         "An unrelated question"
#     ]
#     for q in test_queries:
#         print(f">>> {q}")
#         print(handle_query(q))
#         print()
# import spacy
# from textblob import TextBlob
# from transformers import pipeline
# from crewai import Task, Process

# Import each agent module
# import agents.admission as admission_module
# import agents.program as program_module
# import agents.faculties as faculty_module
# import agents.member as member_module
# import agents.student as student_module
# import agents.campus as campus_module
# import agents.default_agent as default_module

# # 1. Preprocessing: lowercase, spelling correction, punctuation removal
# nlp = spacy.load("en_core_web_sm")

# def preprocess_query(text: str) -> str:
#     corrected = str(TextBlob(text.lower()).correct())
#     doc = nlp(corrected)
#     tokens = [token.text for token in doc if not token.is_punct]
#     return " ".join(tokens)

# # 2. Zero-shot intent classification
# zero_shot = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli",
#     device=0
# )

# # Map intent labels to agent modules
# INTENTS = {
#     "admission": admission_module.admission_agent,
#     "program":  program_module.program_agent,
#     "faculties": faculty_module.faculty_agent,
#     "member":   member_module.board_agent,
#     "student":  student_module.student_agent,
#     "campus":   campus_module.campus_agent,
# }
# INTENT_LABELS = list(INTENTS.keys()) + ["general_fallback"]
# THRESHOLD = 0.4

# # Classifier
# def classify_intent(query: str) -> str:
#     result = zero_shot(query, INTENT_LABELS)
#     label, score = result["labels"][0], result["scores"][0]
#     return label if score >= THRESHOLD else "general_fallback"

# # 3. Setup CrewAI TaskManager with all agents
# # NEW (fixed):
# all_agents = list(INTENTS.values()) + [default_module.default_agent]
# task_manager = Process(all_agents)


# # 4. Main handler

# def handle_query(user_query: str) -> str:
#     clean_q = preprocess_query(user_query)
#     intent = classify_intent(clean_q)
#     if intent != "general_fallback":
#         agent_module = INTENTS[intent]
#         task = Task(
#             name=f"{intent}_query",
#             description=f"Handle '{intent}' user query.",
#             instructions=f"Answer the user query about {intent} using the registered {intent} agent.",
#             expected_output="A natural-language response.",
#             agent=agent_module,
#             async_execution=False
#         )
#         return task_manager.assign_task(task, agent_module)
#     # fallback to default agent
#     return default_module.Default_agent.run(clean_q)

# if __name__ == "__main__":
#     print("Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ("exit", "quit"):
#             break
#         print("Bot:", handle_query(user_input))













#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import multiprocessing as mp

# ─── 1) FORCE 'spawn' FOR CUDA‐SAFE SUBPROCESSING ───────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

# ─── 2) CORE DEPENDENCIES ────────────────────────────────────────────────────
import spacy
from textblob import TextBlob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─── 3) IMPORT YOUR CREWAI AGENTS ────────────────────────────────────────────
from agents.admission  import admission_agent,admission_task
from agents.program    import program_agent, program_task
from agents.faculties  import faculty_agent, faculty_task
from agents.member     import board_agent, board_task
from agents.student    import student_agent,    student_task
from agents.default_agent import default_agent, default_task
from agents.campus     import campus_agent, campus_task

# ─── 4) PREPROCESSING ───────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")
def preprocess_query(text: str) -> str:
    corrected = str(TextBlob(text.lower()).correct())
    doc = nlp(corrected)
    return " ".join(tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_stop)

# ─── 5) EMBEDDINGS & AGENT PROFILES ─────────────────────────────────────────
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

INTENT_AGENTS = {
    "admission": admission_agent,
    "program":   program_agent,
    "faculties": faculty_agent,
    "member":    board_agent,
    "student":   student_agent,
    "campus":    campus_agent,
}
INTENT_PROFILES = {
    label: f"{agent.backstory} {agent.goal}"
    for label, agent in INTENT_AGENTS.items()
}

# Precompute embeddings for each agent profile
INTENT_EMBS = {
    label: np.array(embedder.embed_documents([profile])[0])
    for label, profile in INTENT_PROFILES.items()
}

SIMILARITY_THRESHOLD = 0.3  # tune this as needed

# ─── 6) EMBEDDING‐BASED INTENT CLASSIFICATION ───────────────────────────────
def classify_intent(query: str) -> str:
    q_emb = np.array(embedder.embed_query(query))
    sims = {
        label: cosine_similarity(q_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
        for label, emb in INTENT_EMBS.items()
    }
    best_label, best_score = max(sims.items(), key=lambda kv: kv[1])
    return best_label if best_score >= SIMILARITY_THRESHOLD else "general_fallback"

# ─── 7) MAIN HANDLER WITH FALLBACK TO DEFAULT_AGENT ─────────────────────────

def handle_query(user_query: str) -> str:
    clean_q = preprocess_query(user_query)
    intent  = classify_intent(clean_q)

    # Map each intent to its Task.callback
    HANDLER_FUNCS = {
        "admission": admission_task.callback,
        "program":   program_task.callback,
        "faculties": faculty_task.callback,
        "member":    board_task.callback,
        "student":   student_task.callback,
        "campus":    campus_task.callback,
    }
    # Fallback to the default_task callback
    handler = HANDLER_FUNCS.get(intent, default_task.callback)

    try:
        # wrap the raw string in an object with an .input attribute
        class Q: 
            def __init__(self, i): self.input = i

        result = handler(Q(user_query))

        # if the specialist handler apologizes or returns nothing, delegate to default
        if not result or result.lower().startswith("sorry"):
            return default_task.callback(Q(user_query))

        return result

    except Exception:
        return default_task.callback(Q(user_query))
# ─── 8) OPTIONAL CLI CONCURRENCY WRAPPER ──────────────────────────────────
process_pool = ProcessPoolExecutor(max_workers=4)
thread_pool  = ThreadPoolExecutor(max_workers=20)

def query_in_process(q: str) -> str:
    return process_pool.submit(handle_query, q).result()

if __name__ == "__main__":
    print("DSEU Unified Assistant (type 'exit' or 'quit' to stop')")
    runner = query_in_process

    for line in sys.stdin:
        q = line.strip()
        if q.lower() in ("exit", "quit"):
            break
        print(f"\n🤖 {runner(q)}\n")
