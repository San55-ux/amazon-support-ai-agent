"""
Baselines for comparative benchmarking:
1. Baseline 1 (Trivial Baseline):
   - Majority class intent prediction (ORDER_STATUS_DELIVERY)
   - Static canned response
   - Naive keyword escalation (escalates only if 'fraud' or 'lawyer' is present)

2. Baseline 2 (Simple Baseline):
   - Zero-shot BM25 retrieval without intent classification
   - Uncalibrated heuristic triage (escalates on arbitrary tweet length > 120 chars or 'help')
   - Standard template response without confidence gating or PII redaction
"""
import time
from typing import Dict, Any, List
from src.config import Intent, TriageDecision, EscalationReason, AgentConfig
from src.retriever import HistoricalExemplarRetriever


class TrivialBaselineAgent:
    """
    Baseline 1: Trivial baseline.
    Predicts majority class, uses static canned template, and naive regex.
    """
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.majority_intent = Intent.ORDER_STATUS_DELIVERY.value
        self.canned_reply = (
            "Thanks for reaching out! Please check your order status and tracking online at "
            "https://amzn.to/your-orders. Let us know if you need anything else. ^AMZ"
        )

    def process_tweet(self, customer_tweet: str) -> Dict[str, Any]:
        start = time.perf_counter()
        lower = customer_tweet.lower()

        # Naive keyword escalation
        if "fraud" in lower or "lawyer" in lower:
            decision = TriageDecision.ESCALATE_HUMAN.value
            reason = EscalationReason.SECURITY_FRAUD_RISK.value
            reply = f"Please DM us at https://amzn.to/help-dm. ^AMZ"
        else:
            decision = TriageDecision.AUTO_HANDLE.value
            reason = EscalationReason.NONE_AUTO_HANDLED.value
            reply = self.canned_reply

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "customer_tweet": customer_tweet,
            "predicted_intent": self.majority_intent,
            "intent_confidence": 0.50,
            "triage_decision": decision,
            "escalation_reason": reason,
            "generated_reply": reply,
            "latency_ms": elapsed_ms
        }


class SimpleBaselineAgent:
    """
    Baseline 2: Simple baseline.
    Pure BM25 retrieval to find closest historical query without calibrated intent taxonomy,
    uses heuristic triage (long tweets or 'help' -> escalate), and direct copy of top retrieved response.
    """
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.retriever = HistoricalExemplarRetriever(config=self.config)

    def process_tweet(self, customer_tweet: str) -> Dict[str, Any]:
        start = time.perf_counter()
        lower = customer_tweet.lower()

        # Retrieve top match purely via BM25 lexical match
        matches = self.retriever.retrieve(query=customer_tweet, intent=None, top_k=1)
        top_match = matches[0] if matches else None

        predicted_intent = top_match.get("intent", Intent.GENERAL_INQUIRY_FEEDBACK.value) if top_match else Intent.GENERAL_INQUIRY_FEEDBACK.value

        # Uncalibrated naive triage: escalate if tweet is lengthy or contains exclamation mark
        if len(customer_tweet) > 130 or "!" in customer_tweet or "urgent" in lower:
            decision = TriageDecision.ESCALATE_HUMAN.value
            reason = EscalationReason.HIGH_CUSTOMER_FRUSTRATION.value
            reply = f"We are sorry for the issue. Please send us a DM at https://amzn.to/help-dm. ^AMZ"
        else:
            decision = TriageDecision.AUTO_HANDLE.value
            reason = EscalationReason.NONE_AUTO_HANDLED.value
            reply = top_match.get("historical_reply", "Please check your account at https://amzn.to/your-orders. ^AMZ") if top_match else "Please check https://amzn.to/your-orders. ^AMZ"

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "customer_tweet": customer_tweet,
            "predicted_intent": predicted_intent,
            "intent_confidence": 0.65,
            "triage_decision": decision,
            "escalation_reason": reason,
            "generated_reply": reply,
            "latency_ms": elapsed_ms
        }
