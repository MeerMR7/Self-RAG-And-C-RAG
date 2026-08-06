from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your graph/function from your existing rag_pipeline.py script
# (Make sure 'app_graph' matches the variable name of your compiled graph in rag_pipeline.py)
from rag_pipeline import app_graph  

app = FastAPI(title="Self-RAG & CRAG Backend")

# 🌐 Allow your live Vercel frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-dashboard-nine.vercel.app",  # Your live Next.js app
        "http://localhost:3000",                  # Local Next.js testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryPayload(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(payload: QueryPayload):
    # Pass input query directly to your LangGraph logic in rag_pipeline.py
    initial_state = {"question": payload.query}
    result = app_graph.invoke(initial_state)
    
    # Return the AI response back to Next.js
    return {
        "status": "success",
        "answer": result.get("generation", "No answer generated.")
    }

@app.get("/")
def health_check():
    return {"message": "FastAPI AI Backend is live!"}
