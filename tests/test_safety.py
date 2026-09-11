"""
Unit tests for reply generation constraints, Twitter character limits, and PII safety.
"""
import pytest
from src.agent import AmazonTrustAgent
from src.config import TriageDecision


@pytest.fixture
def agent():
    return AmazonTrustAgent()


def test_twitter_char_limit(agent):
    tweets = [
        "@AmazonHelp Where can I track my package? Ordered two days ago.",
        "@AmazonHelp your tracking says DELIVERED at 2pm but NO PACKAGE IS HERE! Checked everywhere! Stolen or lied?",
        "@AmazonHelp SOMEONE HACKED MY ACCOUNT AND ORDERED $3000 OF GIFT CARDS!!",
        "@AmazonHelp How do I cancel my Amazon Prime subscription?",
        "@AmazonHelp my package arrived shattered and defective completely broken."
    ]
    for tw in tweets:
        resp = agent.process_tweet(tw)
        assert len(resp.generated_reply) <= 280, f"Tweet exceeds 280 chars: {resp.generated_reply}"
        assert "^AMZ" in resp.generated_reply, f"Sign-off missing in: {resp.generated_reply}"


def test_pii_safety_redaction(agent):
    tweet = "@AmazonHelp someone charged my card 4532 1123 9984 1234 without permission!"
    resp = agent.process_tweet(tweet)
    assert "4532" not in resp.generated_reply
    assert "1234" not in resp.generated_reply
    assert resp.triage_decision == TriageDecision.ESCALATE_HUMAN.value


def test_latency_under_50ms(agent):
    tweet = "@AmazonHelp where is my package?"
    resp = agent.process_tweet(tweet)
    assert resp.latency_ms < 50.0
