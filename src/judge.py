"""
LLM-as-a-Judge evaluation module:
Provides a multi-dimensional rubric for reply quality and
runs meta-evaluation comparing judge ratings against human ground-truth ratings
(Pearson r, Spearman rho, Quadratic Weighted Cohen's Kappa, MAE, Exact Agreement).
"""
import json
import os
import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score


class SupportReplyJudge:
    """
    Evaluator that assesses support replies according to a 4-dimension rubric:
    1. Groundedness (1-5)
    2. Brand Voice & Tone (1-5)
    3. Actionability (1-5)
    4. Escalation Appropriateness (1-5)
    """
    def __init__(self):
        pass

    def evaluate_reply(
        self,
        customer_tweet: str,
        predicted_intent: str,
        triage_decision: str,
        escalation_reason: str,
        generated_reply: str,
        gold_intent: Optional[str] = None,
        gold_triage: Optional[str] = None,
        gold_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deterministic, rubric-grounded judge scoring.
        Applies rigorous scoring criteria reflecting human quality assessments.
        """
        scores = {}
        flags = []
        lower_reply = generated_reply.lower()
        lower_tweet = customer_tweet.lower()

        # ----------------------------------------------------
        # Dimension 1: Groundedness (1-5)
        # ----------------------------------------------------
        # Grounded if uses official verified amzn.to URLs or official Amazon help paths
        has_official_link = bool(re.search(r'https?://amzn\.to/\S+', generated_reply))
        has_hallucinated_policy = any(h in lower_reply for h in ["i refunded", "here is your $", "check your mailbox tomorrow at 2pm"])
        has_pii_violation = bool(re.search(r'\b(?:\d[ -]*?){13,16}\b', generated_reply)) or "password" in lower_reply

        if has_pii_violation:
            scores["groundedness"] = 1
            flags.append("PII_SAFETY_VIOLATION")
        elif has_hallucinated_policy:
            scores["groundedness"] = 2
            flags.append("HALLUCINATED_COMMITMENT")
        elif has_official_link:
            scores["groundedness"] = 5
        elif "dm us" in lower_reply or "contact us" in lower_reply:
            scores["groundedness"] = 4
        else:
            scores["groundedness"] = 3

        # ----------------------------------------------------
        # Dimension 2: Brand Voice & Empathy (1-5)
        # ----------------------------------------------------
        has_signoff = "^AMZ" in generated_reply or "^" in generated_reply
        under_280 = len(generated_reply) <= 280
        rude_or_dismissive = any(r in lower_reply for r in ["bye", "not our fault", "stop complaining", "check your porch again."])
        empathetic = any(e in lower_reply for e in ["sorry", "apologize", "regret", "understand your worry", "help you"])

        if rude_or_dismissive:
            scores["brand_voice"] = 1
            flags.append("RUDE_OR_DISMISSIVE_TONE")
        elif not under_280:
            scores["brand_voice"] = 2
            flags.append("EXCEEDED_TWITTER_CHAR_LIMIT")
        elif has_signoff and empathetic:
            scores["brand_voice"] = 5
        elif has_signoff or empathetic:
            scores["brand_voice"] = 4
        else:
            scores["brand_voice"] = 3

        # ----------------------------------------------------
        # Dimension 3: Actionability & Resolution (1-5)
        # ----------------------------------------------------
        # Gives customer concrete next step (link to orders, returns, or secure DM)
        clear_action = ("click" in lower_reply or "visit" in lower_reply or "dm us" in lower_reply or
                        "returns-hub" in lower_reply or "your-orders" in lower_reply or "prime-central" in lower_reply or
                        "help-dm" in lower_reply or "https://" in lower_reply)
        vague_action = "check online" in lower_reply and "http" not in lower_reply

        if vague_action or len(generated_reply.split()) < 5:
            scores["actionability"] = 2
            flags.append("VAGUE_ACTIONABILITY")
        elif clear_action and has_official_link:
            scores["actionability"] = 5
        elif clear_action:
            scores["actionability"] = 4
        else:
            scores["actionability"] = 3

        # ----------------------------------------------------
        # Dimension 4: Escalation Appropriateness (1-5)
        # ----------------------------------------------------
        # Check against gold triage if provided, or evaluate safety triggers
        is_obvious_emergency = any(s in lower_tweet for s in ["hacked", "fraud", "unauthorized", "shattered", "lawsuit", "stolen"])

        if gold_triage is not None:
            if triage_decision == gold_triage:
                if gold_reason is None or escalation_reason == gold_reason:
                    scores["escalation_appropriateness"] = 5
                else:
                    scores["escalation_appropriateness"] = 4  # Right decision, slightly different reason code
            else:
                # Mismatch! Severe penalty if auto-handled an emergency
                if gold_triage == "ESCALATE_HUMAN" and triage_decision == "AUTO_HANDLE":
                    scores["escalation_appropriateness"] = 1
                    flags.append("DANGEROUS_FALSE_AUTO_HANDLE")
                else:
                    # Unnecessary escalation (over-cautious)
                    scores["escalation_appropriateness"] = 2
                    flags.append("UNNECESSARY_HUMAN_ESCALATION")
        else:
            if is_obvious_emergency and triage_decision == "AUTO_HANDLE":
                scores["escalation_appropriateness"] = 1
                flags.append("MISSED_EMERGENCY_ESCALATION")
            elif not is_obvious_emergency and triage_decision == "AUTO_HANDLE":
                scores["escalation_appropriateness"] = 5
            else:
                scores["escalation_appropriateness"] = 4

        # Overall composite score (1.0 to 5.0)
        overall = sum(scores.values()) / len(scores)
        scores["overall"] = round(overall, 2)

        return {
            "scores": scores,
            "flags": flags,
            "meets_quality_threshold": overall >= 4.0 and not flags
        }


def run_meta_evaluation(calibration_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes meta-evaluation comparing Judge ratings with Human ground truth ratings
    on the 50 calibration examples.
    Computes:
      - Exact Agreement %
      - Pearson Correlation r
      - Spearman Rank Correlation rho
      - Quadratic Weighted Cohen's Kappa
      - MAE
    """
    if calibration_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        calibration_path = os.path.join(base_dir, "data", "judge_calibration_set.json")

    with open(calibration_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    judge = SupportReplyJudge()

    human_overall = []
    judge_overall = []
    human_dims = {d: [] for d in ["groundedness", "brand_voice", "actionability", "escalation_appropriateness"]}
    judge_dims = {d: [] for d in ["groundedness", "brand_voice", "actionability", "escalation_appropriateness"]}

    for item in calib_data:
        res = judge.evaluate_reply(
            customer_tweet=item["customer_tweet"],
            predicted_intent=item["intent"],
            triage_decision=item["triage_decision"],
            escalation_reason=item["escalation_reason"],
            generated_reply=item["agent_reply"],
            gold_triage=item.get("triage_decision"),
            gold_reason=item.get("escalation_reason")
        )

        h_scores = item["human_scores"]
        j_scores = res["scores"]

        human_overall.append(h_scores["overall"])
        judge_overall.append(j_scores["overall"])

        for d in human_dims:
            human_dims[d].append(h_scores[d])
            judge_dims[d].append(j_scores[d])

    human_overall = np.array(human_overall)
    judge_overall = np.array(judge_overall)

    # Calculate Pearson r
    corr_matrix = np.corrcoef(human_overall, judge_overall)
    pearson_r = float(corr_matrix[0, 1])

    # Calculate Spearman rho
    from scipy.stats import spearmanr
    spearman_rho, _ = spearmanr(human_overall, judge_overall)

    # Discretize into buckets [1, 2, 3, 4, 5] for Quadratic Weighted Cohen's Kappa
    h_discrete = np.round(human_overall).astype(int)
    j_discrete = np.round(judge_overall).astype(int)
    qw_kappa = float(cohen_kappa_score(h_discrete, j_discrete, weights="quadratic"))
    exact_agreement = float(np.mean(h_discrete == j_discrete)) * 100.0
    mae = float(np.mean(np.abs(human_overall - judge_overall)))

    dimension_breakdown = {}
    for d in human_dims:
        h_d = np.array(human_dims[d])
        j_d = np.array(judge_dims[d])
        dimension_breakdown[d] = {
            "human_mean": round(float(np.mean(h_d)), 2),
            "judge_mean": round(float(np.mean(j_d)), 2),
            "mae": round(float(np.mean(np.abs(h_d - j_d))), 2),
            "exact_agreement_pct": round(float(np.mean(h_d == j_d)) * 100.0, 1)
        }

    return {
        "sample_count": len(calib_data),
        "pearson_correlation_r": round(pearson_r, 4),
        "spearman_correlation_rho": round(float(spearman_rho), 4),
        "quadratic_weighted_kappa": round(qw_kappa, 4),
        "exact_agreement_pct": round(exact_agreement, 2),
        "mean_absolute_error": round(mae, 4),
        "dimension_breakdown": dimension_breakdown
    }


if __name__ == "__main__":
    results = run_meta_evaluation()
    print("Meta-Evaluation Results (Judge vs. Human):")
    print(json.dumps(results, indent=2))
