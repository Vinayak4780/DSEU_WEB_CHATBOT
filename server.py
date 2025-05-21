from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import handle_query

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Create(BaseModel):
    prompt: str

@app.post("/prompt")
def create(post: Create):
    response = handle_query(post.prompt)
    return {"response": response}

@app.get("/")
def start():
    return {"message": "Initiating ChatBot....."}
