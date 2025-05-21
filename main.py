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
import spacy
from textblob import TextBlob
from transformers import pipeline

# Import each agent module (handlers defined there)
import agents.admission       as admission_module
import agents.program         as program_module
import agents.faculties       as faculty_module
import agents.member          as member_module
import agents.student         as student_module
import agents.campus          as campus_module
import agents.default_agent   as default_module

# Preprocessing: lowercase, spelling correction, punctuation removal
nlp = spacy.load("en_core_web_sm")
def preprocess_query(text: str) -> str:
    corrected = str(TextBlob(text.lower()).correct())
    doc = nlp(corrected)
    tokens = [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_stop]
    return " ".join(tokens)

# Zero‐shot intent classification
zero_shot = pipeline(
    "zero-shot-classification",  
    model="facebook/bart-large-mnli",
    device=0  # or -1 for CPU
)

# Map intent labels to handler functions
INTENT_HANDLERS = {
    "admission": admission_module.main_bot,
    "program":   program_module.syllabus_bot,
    "faculties": faculty_module.faculty_bot,
    "member":    member_module.board_bot,
    "student":   student_module.student_bot,
    "campus":    campus_module.syllabus_bot,
}
INTENT_LABELS = list(INTENT_HANDLERS.keys()) + ["general_fallback"]
THRESHOLD = 0.4

def classify_intent(query: str) -> str:
    result = zero_shot(query, INTENT_LABELS)
    label, score = result["labels"][0], result["scores"][0]
    return label if score >= THRESHOLD else "general_fallback"

# Main handler without TaskManager
def handle_query(user_query: str) -> str:
    # Preprocess and classify
    clean_q = preprocess_query(user_query)
    intent  = classify_intent(clean_q)

    # Choose corresponding handler or default
    handler = INTENT_HANDLERS.get(intent, default_module.document_qa)
    try:
        return handler(user_query)
    except Exception:
        return "Sorry, I couldn't process that right now."

if __name__ == "__main__":
    print("DSEU Unified Assistant (type 'exit' or 'quit' to stop')")
    for line in sys.stdin:
        q = line.strip()
        if q.lower() in ("exit", "quit"):
            break
        print("\n🤖", handle_query(q), "\n")
