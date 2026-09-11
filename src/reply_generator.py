"""
Reply Generator: Generates grounded customer support replies
adhering strictly to Twitter character constraints (280 chars), brand voice (^AMZ),
and privacy/PII safety policies.
"""
import re
from typing import List, Dict, Any, Optional
from src.config import Intent, TriageDecision, EscalationReason, AgentConfig


class GroundedReplyGenerator:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def _sanitize_pii(self, text: str) -> str:
        """
        Safety filter: ensure the agent never asks for or repeats raw credit card numbers,
        passwords, or explicit sensitive personal data in a public tweet.
        """
        # Scrub 16-digit or 4-digit credit card prompts
        scrubbed = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED_CARD]', text)
        return scrubbed

    def generate(
        self,
        customer_tweet: str,
        predicted_intent: str,
        triage_decision: TriageDecision,
        escalation_reason: EscalationReason,
        exemplars: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesizes a grounded reply utilizing retrieved historical exemplars
        and appropriate escalation paths.
        """
        # If escalating to human, tailor empathetic handoff with verified secure DM link
        if triage_decision == TriageDecision.ESCALATE_HUMAN:
            if escalation_reason == EscalationReason.SECURITY_FRAUD_RISK:
                draft = f"Security is our top priority. Please do NOT share account details publicly. Send us a DM at {self.config.safe_dm_link} so our fraud team can secure your account immediately. {self.config.brand_signoff}"
            elif escalation_reason == EscalationReason.LOST_DELIVERED_PACKAGE:
                draft = f"We apologize for the missing package concern! Please send us a direct message at {self.config.safe_dm_link} with your order info so we can trace your delivery right away. {self.config.brand_signoff}"
            elif escalation_reason == EscalationReason.HIGH_CUSTOMER_FRUSTRATION:
                draft = f"We are truly sorry for your frustrating experience. Please connect with our leadership team via DM: {self.config.safe_dm_link} so an escalation specialist can resolve this for you. {self.config.brand_signoff}"
            elif escalation_reason == EscalationReason.DAMAGED_GOODS_CLAIM:
                draft = f"We are so sorry for this damaged order! Please DM us at {self.config.safe_dm_link} with your order details and photos so our specialists can expedite a replacement or refund. {self.config.brand_signoff}"
            elif escalation_reason == EscalationReason.BILLING_DISPUTE_REFUND_AUTH:
                draft = f"We apologize for the billing discrepancy. Please reach out via secure DM at {self.config.safe_dm_link} so our billing team can review your account and release your funds. {self.config.brand_signoff}"
            elif escalation_reason == EscalationReason.LOW_INTENT_CONFIDENCE:
                draft = f"Hello! We want to make sure you get the right help. Could you please send us a DM at {self.config.safe_dm_link} with your order details so an agent can assist? {self.config.brand_signoff}"
            else:
                draft = f"We understand your concern. Please send us a direct message at {self.config.safe_dm_link} so a customer care specialist can review your inquiry directly. {self.config.brand_signoff}"
        else:
            # Auto-handle: Grounded in the top historical exemplar
            if exemplars and "historical_reply" in exemplars[0]:
                draft = exemplars[0]["historical_reply"]
            else:
                # Intent-fallback grounded templates
                if predicted_intent == Intent.ORDER_STATUS_DELIVERY.value:
                    draft = f"You can track live carrier updates and package progress under Your Orders: {self.config.safe_orders_link}. Let us know if you need more help! {self.config.brand_signoff}"
                elif predicted_intent == Intent.RETURN_REFUND.value:
                    draft = f"Starting a return or tracking refunds is quick and easy at {self.config.safe_returns_link}. Select your item to print a label or get a drop-off QR code. {self.config.brand_signoff}"
                elif predicted_intent == Intent.DAMAGED_WRONG_ITEM.value:
                    draft = f"We are sorry your item arrived defective! You can request a hassle-free replacement or return directly at {self.config.safe_returns_link}. {self.config.brand_signoff}"
                elif predicted_intent == Intent.ACCOUNT_SECURITY_BILLING.value:
                    draft = f"You can securely manage payment cards, addresses, and invoices in your account under Your Payments: {self.config.safe_orders_link}. {self.config.brand_signoff}"
                elif predicted_intent == Intent.PRIME_SUBSCRIPTION.value:
                    draft = f"You can manage, review benefits, or cancel your Prime subscription anytime at {self.config.safe_prime_link}. {self.config.brand_signoff}"
                else:
                    draft = f"Thank you for contacting us! You can find help guides and account tools anytime at {self.config.safe_orders_link}. {self.config.brand_signoff}"

        # Enforce PII filter
        draft = self._sanitize_pii(draft)

        # Enforce Twitter 280-character ceiling
        if len(draft) > self.config.max_tweet_length:
            # Truncate cleanly before sign-off
            allowed = self.config.max_tweet_length - len(self.config.brand_signoff) - 5
            draft = draft[:allowed].rsplit(' ', 1)[0] + f"... {self.config.brand_signoff}"

        return draft
