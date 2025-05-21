from langchain_community.llms import LlamaCpp

LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
shared_llm = LlamaCpp(
    model_path=LLM_PATH,
    n_ctx=8192,
    temperature=0.2,
    max_tokens=10000,
    n_gpu_layers=100
)