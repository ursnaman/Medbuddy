from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conversation import MedBuddyChat

app = FastAPI(
    title="MedBuddy API",
    description="AI Medical Assistant powered by LLaMA 3",
    version="1.0.0"
)

conversations = {}


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    reply: str
    user_id: str
    total_messages: int


@app.get("/")
def home():
    return {
        "status": "MedBuddy is running!",
        "message": "Visit /docs to test the API"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        if request.user_id not in conversations:
            conversations[request.user_id] = MedBuddyChat(
                user_id=request.user_id
            )

        user_chat = conversations[request.user_id]
        ai_reply = user_chat.chat(request.message)

        return ChatResponse(
            reply=ai_reply,
            user_id=request.user_id,
            total_messages=len(user_chat.conversation_history)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/clear/{user_id}")
def clear(user_id: str):
    if user_id in conversations:
        conversations[user_id].clear_history()
        del conversations[user_id]
    return {"message": f"Conversation cleared for {user_id}"}


@app.get("/history/{user_id}")
def history(user_id: str):
    if user_id not in conversations:
        return {"history": [], "total": 0}
    h = conversations[user_id].conversation_history
    return {
        "user_id": user_id,
        "total": len(h),
        "history": h
    }


if __name__ == "__main__":
    print("Starting MedBuddy API...")
    print("Visit: http://localhost:8000/docs")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )