"""
AmazonTrustAgent: Unified pipeline orchestrating intent classification,
retrieval of historical brand resolutions, safety-critical triage, and reply generation.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.config import AgentConfig, TriageDecision, EscalationReason
from src.intent_classifier import CalibratedIntentClassifier
from src.retriever import HistoricalExemplarRetriever
from src.triage_engine import TriageEngine
from src.reply_generator import GroundedReplyGenerator


@dataclass
class AgentResponse:
    customer_tweet: str
    predicted_intent: str
    intent_confidence: float
    all_intent_probabilities: Dict[str, float]
    triage_decision: str
    escalation_reason: str
    risk_score: float
    audit_triggers: List[str]
    retrieved_exemplars: List[Dict[str, Any]]
    generated_reply: str
    character_count: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_tweet": self.customer_tweet,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "all_intent_probabilities": {k: round(v, 4) for k, v in self.all_intent_probabilities.items()},
            "triage_decision": self.triage_decision,
            "escalation_reason": self.escalation_reason,
            "risk_score": round(self.risk_score, 4),
            "audit_triggers": self.audit_triggers,
            "retrieved_exemplars_count": len(self.retrieved_exemplars),
            "generated_reply": self.generated_reply,
            "character_count": self.character_count,
            "latency_ms": round(self.latency_ms, 2)
        }


class AmazonTrustAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.classifier = CalibratedIntentClassifier(config=self.config)
        self.retriever = HistoricalExemplarRetriever(config=self.config)
        self.triage = TriageEngine(config=self.config)
        self.generator = GroundedReplyGenerator(config=self.config)

    def process_tweet(self, customer_tweet: str) -> AgentResponse:
        start_time = time.perf_counter()

        # 1. Intent Classification with calibrated confidence
        intent, conf, all_probs, is_low_conf = self.classifier.predict(customer_tweet)

        # 2. Multi-Stage Escalation Triage
        decision, reason, risk_score, triggers = self.triage.evaluate(
            customer_tweet=customer_tweet,
            predicted_intent=intent,
            intent_confidence=conf,
            is_low_confidence=is_low_conf
        )

        # 3. Grounded Retrieval
        exemplars = self.retriever.retrieve(
            query=customer_tweet,
            intent=intent,
            top_k=self.config.retrieval_top_k
        )

        # 4. Reply Generation & Guardrail Verification
        reply = self.generator.generate(
            customer_tweet=customer_tweet,
            predicted_intent=intent,
            triage_decision=decision,
            escalation_reason=reason,
            exemplars=exemplars
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            customer_tweet=customer_tweet,
            predicted_intent=intent,
            intent_confidence=conf,
            all_intent_probabilities=all_probs,
            triage_decision=decision.value,
            escalation_reason=reason.value,
            risk_score=risk_score,
            audit_triggers=triggers,
            retrieved_exemplars=exemplars,
            generated_reply=reply,
            character_count=len(reply),
            latency_ms=elapsed_ms
        )
