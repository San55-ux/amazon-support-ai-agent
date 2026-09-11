"""
Unit tests for CalibratedIntentClassifier.
"""
import pytest
from src.config import Intent
from src.intent_classifier import CalibratedIntentClassifier


@pytest.fixture
def classifier():
    return CalibratedIntentClassifier()


def test_order_status_intent(classifier):
    text = "@AmazonHelp where is my package tracking number? It has been 4 days."
    intent, conf, probs, is_low = classifier.predict(text)
    assert intent == Intent.ORDER_STATUS_DELIVERY.value
    assert conf >= 0.60
    assert not is_low


def test_return_refund_intent(classifier):
    text = "@AmazonHelp how do I print a return label or drop off at Kohl's?"
    intent, conf, probs, is_low = classifier.predict(text)
    assert intent == Intent.RETURN_REFUND.value
    assert conf >= 0.60


def test_damaged_item_intent(classifier):
    text = "@AmazonHelp my package arrived shattered and broken into glass pieces."
    intent, conf, probs, is_low = classifier.predict(text)
    assert intent == Intent.DAMAGED_WRONG_ITEM.value
    assert conf >= 0.60


def test_security_fraud_intent(classifier):
    text = "@AmazonHelp someone hacked my account and there are unauthorized charges on my card!"
    intent, conf, probs, is_low = classifier.predict(text)
    assert intent == Intent.ACCOUNT_SECURITY_BILLING.value
    assert conf >= 0.70


def test_prime_subscription_intent(classifier):
    text = "@AmazonHelp how do I cancel my Amazon Prime membership?"
    intent, conf, probs, is_low = classifier.predict(text)
    assert intent == Intent.PRIME_SUBSCRIPTION.value
    assert conf >= 0.60


def test_general_inquiry_intent(classifier):
    text = "@AmazonHelp does Amazon ship Kindle devices internationally to Australia?"
    intent, conf, probs, is_low = classifier.predict(text)
    assert intent in [Intent.GENERAL_INQUIRY_FEEDBACK.value, Intent.ORDER_STATUS_DELIVERY.value]
