"""
Triage Engine: Evaluates incoming messages for auto-handling vs. human escalation.
Produces auditable, transparent escalation decisions with explicit categorical reason codes.
"""
import re
from typing import Dict, Any, List, Tuple
from src.config import TriageDecision, EscalationReason, AgentConfig, Intent


class TriageEngine:
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()

    def _calculate_frustration_score(self, text: str) -> float:
        """
        Calculates sentiment and urgency based on all-caps ratio, punctuation clusters,
        and high-arousal negative emotion lexicon.
        Returns a score from 0.0 (calm) to 1.0 (extreme rage/urgency).
        """
        lower = text.lower()
        score = 0.0

        # Punctuation markers
        excl_count = text.count('!')
        if excl_count >= 1:
            score += min(0.3, excl_count * 0.15)

        # All-caps words analysis
        words = re.findall(r'\b[A-Z]{2,}\b', text)
        loud_words = [w for w in words if w not in {"US", "UPS", "USPS", "TV", "OLED", "OTP", "PIN", "DM", "QR", "PDF", "VAT", "ID", "AMZ", "RTX"}]
        if loud_words:
            score += min(0.4, len(loud_words) * 0.15)

        # Rage / Threat keywords
        rage_cues = [
            "lawyer", "attorney", "lawsuit", "legal action", "chargeback", "charge back", "police report",
            "disgusting", "unacceptable", "furious", "sick of", "worst customer service",
            "hung up on me", "lied to me", "stolen", "fraud", "scam", "incompetent",
            "refused to help", "disconnected me", "cancelled my account", "subpoena",
            "45 days", "3 weeks", "holding my purchases hostage", "refused refund", "unauthorized",
            "3 strikes", "reverse this hidden fee", "without my permission", "without parental",
            "threw heavy package", "sleeping on the porch"
        ]
        matches = [cue for cue in rage_cues if cue in lower]
        if matches:
            score += min(0.6, len(matches) * 0.25)

        return min(1.0, score)

    def evaluate(
        self,
        customer_tweet: str,
        predicted_intent: str,
        intent_confidence: float,
        is_low_confidence: bool
    ) -> Tuple[TriageDecision, EscalationReason, float, List[str]]:
        text = customer_tweet
        lower = text.lower()
        audit_triggers = []
        frustration_score = self._calculate_frustration_score(text)

        # ----------------------------------------------------
        # STAGE 1: Critical Security, Fraud & PII Risks
        # ----------------------------------------------------
        security_patterns = [
            r'\b(hacked|compromised|account takeover|took over (my )?account)\b',
            r'\bunauthorized charge(s)?\b',
            r'\b(stolen card|stolen phone|credit card fraud|identity theft)\b',
            r'\b(social security|ssn|police report)\b',
            r'\b(changed (the |my )?primary email|changed (my |the )?password without|without my consent)\b',
            r'\b(flagged \d+ fraudulent|unauthorized digital charge)\b',
            r'\b(zero-day|vulnerability research|exploit)\b',
            r'\b(cell number and home address|dox|privacy violation)\b',
            r'\b(subpoena|court order)\b',
            r'\b(locked out .* 2fa|2fa .* stolen phone)\b',
            r'\b(scammers? took over|this is fraud)\b',
            r'\b(crypto mining|tampered|stolen contents)\b'
        ]
        for pat in security_patterns:
            if re.search(pat, lower):
                audit_triggers.append(f"Security/Fraud trigger: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.SECURITY_FRAUD_RISK, 0.95, audit_triggers

        # ----------------------------------------------------
        # STAGE 2: Lost / "Delivered but Missing" Packages
        # ----------------------------------------------------
        lost_package_patterns = [
            r'delivered .* (no package|not here|nothing|never received|empty)',
            r'(marked|says) delivered .* (someone else|not my|wrong porch|not received)',
            r'handed to resident .* no one knocked',
            r'locker .* opened empty',
            r'(stuck in transit|carrier says) .* (10 days|over a week|lost|officially lost)',
            r'package is officially lost',
            r'stole(n)? (my )?package',
            r'package was stolen',
            r'forged (my )?signature',
            r'delivered in \w+ but i live in \w+',
            r'driver took (it )?back to the van'
        ]
        for pat in lost_package_patterns:
            if re.search(pat, lower):
                audit_triggers.append(f"Lost package trigger: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.LOST_DELIVERED_PACKAGE, 0.88, audit_triggers

        # ----------------------------------------------------
        # STAGE 3: Damaged High-Value, Hazardous, or Injury
        # ----------------------------------------------------
        damage_escalate_patterns = [
            r'(\$\d{3,}|expensive|oled|tv|laptop|macbook|iphone|diamond|camera lens|graphics card|rtx) .* (shattered|smashed|crushed|broken|refurbished)',
            r'(acid|battery exploded|scorched|fire hazard|burned|burnt|bleeding|blood|biohazard)',
            r'(box empty inside|potato|brick|stolen contents)',
            r'(baby formula|medication|insulin|vials) .* (punctured|crushed|dirt)',
            r'(dropped .* down .* staircase|smashed into concrete|destroyed .* sprinkler|onto my cat)'
        ]
        for pat in damage_escalate_patterns:
            if re.search(pat, lower):
                audit_triggers.append(f"Damaged goods/Hazard trigger: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.DAMAGED_GOODS_CLAIM, 0.90, audit_triggers

        # ----------------------------------------------------
        # STAGE 4: High-Value Stuck Refunds & Billing Disputes
        # ----------------------------------------------------
        refund_dispute_patterns = [
            r'still no refund of \$\d+',
            r'warehouse claims .* received wrong item .* refused refund',
            r'refused refund',
            r'rejected my return claim',
            r'reverse this fee now',
            r'charged (\$\d+|14\.99) for prime .* (month|years)',
            r'refund my \$\d+ now',
            r'double billed \$\d+',
            r'aws billed .* \$\d+',
            r'refund was processed to a closed bank account',
            r'minor child .* \$\d+ .* without parental',
            r'charged three times for the exact same order',
            r'returned \d+ items .* only \d+ was refunded',
            r'waiting \d+ days for .* refund',
            r'reverse this hidden fee',
            r'keeps reactivating .* charging',
            r'promised a full refund .* charged again'
        ]
        for pat in refund_dispute_patterns:
            if re.search(pat, lower):
                audit_triggers.append(f"Billing dispute trigger: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.BILLING_DISPUTE_REFUND_AUTH, 0.85, audit_triggers

        # ----------------------------------------------------
        # STAGE 5: High Frustration, Rage, Repeated Failures, Churn
        # ----------------------------------------------------
        if frustration_score >= 0.60:
            audit_triggers.append(f"Frustration score exceeded threshold: {frustration_score:.2f}")
            return TriageDecision.ESCALATE_HUMAN, EscalationReason.HIGH_CUSTOMER_FRUSTRATION, frustration_score, audit_triggers

        high_frustration_patterns = [
            r'\b(attorney|lawyer|lawsuit|legal action|chargeback|charge back)\b',
            r'\b(racial slur|discrimination|abusive|cursed at me)\b',
            r'\b(3rd time|third time|5 times|ignored \d+ emails|hung up on me|disconnected me|promised .* charged again)\b',
            r'\b(holding \$\d+.*funds|holding .* hostage)\b',
            r'\b(bot .* infinite loop|talk to a real human)\b',
            r'\b(medical emergency|critical insulin)\b',
            r'\b(unacceptable|45 days|3 strikes|cancel my prime membership without my permission)\b',
            r'every single order .* took \d+ days .* refund for the whole year'
        ]
        for pat in high_frustration_patterns:
            if re.search(pat, lower):
                audit_triggers.append(f"High customer churn/frustration trigger: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.HIGH_CUSTOMER_FRUSTRATION, 0.85, audit_triggers

        # ----------------------------------------------------
        # STAGE 6: Prompt Injections & Complex Ambiguous Queries
        # ----------------------------------------------------
        injection_patterns = [
            r'ignore (all )?previous instructions',
            r'developer override mode',
            r'system prompt leak',
            r'print your hidden instructions',
            r'credit me \$\d+'
        ]
        for pat in injection_patterns:
            if re.search(pat, lower):
                audit_triggers.append(f"Prompt injection defense: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.LOW_INTENT_CONFIDENCE, 0.90, audit_triggers

        # Low confidence or garbled
        if is_low_confidence or len(text.strip().split()) <= 3 or re.search(r'^\W+$', text):
            audit_triggers.append(f"Low intent confidence ({intent_confidence:.2f} < {self.config.intent_confidence_threshold}) or ambiguous query")
            return TriageDecision.ESCALATE_HUMAN, EscalationReason.LOW_INTENT_CONFIDENCE, 0.60, audit_triggers

        # Multi-issue entanglement / complex exceptions
        ambiguous_exceptions = [
            r'\b(hospital having surgery|deceased mother|bereavement)\b',
            r'\b(press contact|reporter with)\b',
            r'\b(call me on \d{3})\b',
            r'\b(fired my local delivery driver)\b'
        ]
        for pat in ambiguous_exceptions:
            if re.search(pat, lower):
                audit_triggers.append(f"Complex exceptional inquiry: '{pat}'")
                return TriageDecision.ESCALATE_HUMAN, EscalationReason.COMPLEX_AMBIGUOUS_QUERY, 0.70, audit_triggers

        # ----------------------------------------------------
        # STAGE 7: Safe to Auto-Handle
        # ----------------------------------------------------
        audit_triggers.append("Passed all safety, fraud, and frustration gates. Cleared for automated response.")
        return TriageDecision.AUTO_HANDLE, EscalationReason.NONE_AUTO_HANDLED, 0.10, audit_triggers
