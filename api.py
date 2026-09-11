"""
FastAPI production REST API for @AmazonHelp Customer Support AI Agent.
Exposes endpoints for real-time inference, batch evaluation, and health checks.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import AmazonTrustAgent
from src.config import Intent, TriageDecision, EscalationReason

app = FastAPI(
    title="@AmazonHelp Support AI Agent API",
    description="Production REST API for classifying customer support tweets, retrieving grounded resolutions, and automated safety triage.",
    version="1.0.0"
)

agent = AmazonTrustAgent()


class TweetRequest(BaseModel):
    customer_tweet: str = Field(
        ...,
        description="The incoming tweet message from the customer",
        example="@AmazonHelp where is my package tracking number? It has been 4 days."
    )


class TweetResponse(BaseModel):
    customer_tweet: str
    predicted_intent: str
    intent_confidence: float
    all_intent_probabilities: Dict[str, float]
    triage_decision: str
    escalation_reason: str
    risk_score: float
    audit_triggers: List[str]
    retrieved_exemplars_count: int
    generated_reply: str
    character_count: int
    latency_ms: float


class BatchTweetRequest(BaseModel):
    tweets: List[str]


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "@AmazonHelp Customer Support AI Agent API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/process", response_model=TweetResponse, tags=["Inference"])
def process_tweet_endpoint(request: TweetRequest):
    if not request.customer_tweet.strip():
        raise HTTPException(status_code=400, detail="Customer tweet cannot be empty.")
    response = agent.process_tweet(request.customer_tweet)
    return response.to_dict()


@app.post("/batch", response_model=List[TweetResponse], tags=["Inference"])
def batch_process_endpoint(request: BatchTweetRequest):
    results = []
    for tw in request.tweets:
        if tw.strip():
            res = agent.process_tweet(tw)
            results.append(res.to_dict())
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
