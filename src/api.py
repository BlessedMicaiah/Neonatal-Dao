from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from inference import InferenceEngine

app = FastAPI(title="Neonatal Dao API", version="0.1.0")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# Load model once at startup
engine = InferenceEngine()


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Neonatal Dao API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = engine.generate(request.question)
    return ChatResponse(answer=answer)
