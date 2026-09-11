"""
Configuration and constants for the @AmazonHelp Customer Support AI Agent.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Intent(str, Enum):
    ORDER_STATUS_DELIVERY = "ORDER_STATUS_DELIVERY"
    RETURN_REFUND = "RETURN_REFUND"
    DAMAGED_WRONG_ITEM = "DAMAGED_WRONG_ITEM"
    ACCOUNT_SECURITY_BILLING = "ACCOUNT_SECURITY_BILLING"
    PRIME_SUBSCRIPTION = "PRIME_SUBSCRIPTION"
    GENERAL_INQUIRY_FEEDBACK = "GENERAL_INQUIRY_FEEDBACK"


class TriageDecision(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"


class EscalationReason(str, Enum):
    NONE_AUTO_HANDLED = "NONE_AUTO_HANDLED"
    SECURITY_FRAUD_RISK = "SECURITY_FRAUD_RISK"
    LOST_DELIVERED_PACKAGE = "LOST_DELIVERED_PACKAGE"
    HIGH_CUSTOMER_FRUSTRATION = "HIGH_CUSTOMER_FRUSTRATION"
    DAMAGED_GOODS_CLAIM = "DAMAGED_GOODS_CLAIM"
    BILLING_DISPUTE_REFUND_AUTH = "BILLING_DISPUTE_REFUND_AUTH"
    LOW_INTENT_CONFIDENCE = "LOW_INTENT_CONFIDENCE"
    COMPLEX_AMBIGUOUS_QUERY = "COMPLEX_AMBIGUOUS_QUERY"


INTENT_DESCRIPTIONS: Dict[str, str] = {
    Intent.ORDER_STATUS_DELIVERY.value: "Tracking shipments, transit delays, carrier issues, or packages marked delivered but missing.",
    Intent.RETURN_REFUND.value: "Return process, drop-off locations, return label generation, refund turnaround time.",
    Intent.DAMAGED_WRONG_ITEM.value: "Damaged packaging, broken items, missing accessories, or incorrect product delivered.",
    Intent.ACCOUNT_SECURITY_BILLING.value: "Unauthorized credit card charges, account takeover/lock, phishing, OTP or sign-in issues.",
    Intent.PRIME_SUBSCRIPTION.value: "Amazon Prime membership fees, cancellation, unexpected renewals, Prime Video streaming access.",
    Intent.GENERAL_INQUIRY_FEEDBACK.value: "Product stock availability, website/app glitches, trade-in program, international delivery questions.",
}


@dataclass
class AgentConfig:
    brand_handle: str = "@AmazonHelp"
    brand_signoff: str = "^AMZ"
    max_tweet_length: int = 280
    intent_confidence_threshold: float = 0.58
    frustration_sentiment_threshold: float = -0.40
    retrieval_top_k: int = 3
    safe_dm_link: str = "https://amzn.to/help-dm"
    safe_orders_link: str = "https://amzn.to/your-orders"
    safe_returns_link: str = "https://amzn.to/returns-hub"
    safe_prime_link: str = "https://amzn.to/prime-central"
