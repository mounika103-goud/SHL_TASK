from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.models import ChatRequest, ChatResponse
from app.agent import process_chat

app = FastAPI(title="SHL Conversational Agent API")

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for the SHL conversational agent.
    Takes a stateless conversation history and returns the agent's reply and recommendations.
    """
    try:
        response = process_chat(request.messages)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
