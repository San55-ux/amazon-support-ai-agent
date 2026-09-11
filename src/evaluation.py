"""
Comprehensive evaluation harness:
Runs automated metrics and rubric evaluation on the 200-sample golden dataset
across Baseline 1, Baseline 2, and the Proposed AmazonTrustAgent.
"""
import json
import os
import time
import re
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.config import Intent, TriageDecision
from src.agent import AmazonTrustAgent
from src.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from src.judge import SupportReplyJudge


class BenchmarkEvaluator:
    def __init__(self, golden_path: str = None):
        if golden_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            golden_path = os.path.join(base_dir, "data", "golden_eval_set.json")
        self.golden_path = golden_path
        with open(self.golden_path, "r", encoding="utf-8") as f:
            self.golden_data = json.load(f)
        self.judge = SupportReplyJudge()

    def _compute_token_jaccard(self, str1: str, str2: str) -> float:
        set1 = set(re.findall(r'\w+', str1.lower()))
        set2 = set(re.findall(r'\w+', str2.lower()))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def evaluate_model(self, model_name: str, model_instance) -> Dict[str, Any]:
        """
        Runs the full 200-item golden evaluation set through the given model.
        """
        gold_intents = []
        pred_intents = []
        gold_triages = []
        pred_triages = []
        is_security_flags = []
        latencies = []
        jaccard_scores = []
        pii_violations = 0
        judge_overall_scores = []
        judge_groundedness = []
        judge_brand_voice = []
        judge_actionability = []
        judge_escalation = []

        all_intents_list = [i.value for i in Intent]

        for sample in self.golden_data:
            tweet = sample["customer_tweet"]
            gold_i = sample["intent"]
            gold_t = sample["triage_decision"]
            gold_r = sample["escalation_reason"]
            ref_rep = sample["reference_resolution"]
            is_sec = sample.get("is_security_risk", False) or sample.get("is_lost_package", False) or sample.get("customer_frustration", 0) >= 0.8

            gold_intents.append(gold_i)
            gold_triages.append(gold_t)
            is_security_flags.append(is_sec)

            # Process with model
            res = model_instance.process_tweet(tweet)
            if hasattr(res, "to_dict"):
                res = res.to_dict()

            pred_i = res["predicted_intent"]
            pred_t = res["triage_decision"]
            pred_r = res.get("escalation_reason", "NONE_AUTO_HANDLED")
            rep = res["generated_reply"]
            lat = res["latency_ms"]

            pred_intents.append(pred_i)
            pred_triages.append(pred_t)
            latencies.append(lat)

            # Check PII violation (soliciting card or password)
            if re.search(r'\b(?:\d[ -]*?){13,16}\b', rep) or "password" in rep.lower():
                pii_violations += 1

            # Jaccard overlap with gold reference reply
            jaccard_scores.append(self._compute_token_jaccard(rep, ref_rep))

            # Run judge
            j_eval = self.judge.evaluate_reply(
                customer_tweet=tweet,
                predicted_intent=pred_i,
                triage_decision=pred_t,
                escalation_reason=pred_r,
                generated_reply=rep,
                gold_intent=gold_i,
                gold_triage=gold_t,
                gold_reason=gold_r
            )
            scores = j_eval["scores"]
            judge_overall_scores.append(scores["overall"])
            judge_groundedness.append(scores["groundedness"])
            judge_brand_voice.append(scores["brand_voice"])
            judge_actionability.append(scores["actionability"])
            judge_escalation.append(scores["escalation_appropriateness"])

        # Intent metrics
        intent_acc = float(accuracy_score(gold_intents, pred_intents))
        prec, rec, f1, _ = precision_recall_fscore_support(gold_intents, pred_intents, average="macro", zero_division=0)

        # Escalation metrics
        triage_labels = [TriageDecision.AUTO_HANDLE.value, TriageDecision.ESCALATE_HUMAN.value]
        t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(
            [1 if t == TriageDecision.ESCALATE_HUMAN.value else 0 for t in gold_triages],
            [1 if t == TriageDecision.ESCALATE_HUMAN.value else 0 for t in pred_triages],
            average="binary",
            zero_division=0
        )

        # Critical Safety Analysis: Did we catch all true security/lost/rage crises?
        critical_indices = [idx for idx, sec in enumerate(is_security_flags) if sec]
        total_critical = len(critical_indices)
        critical_escalated = sum(1 for idx in critical_indices if pred_triages[idx] == TriageDecision.ESCALATE_HUMAN.value)
        safety_recall = (critical_escalated / total_critical * 100.0) if total_critical > 0 else 100.0

        # Dangerous false auto-handle rate: critical cases mistakenly auto-handled
        dangerous_false_auto_count = total_critical - critical_escalated
        dangerous_false_auto_rate = (dangerous_false_auto_count / total_critical * 100.0) if total_critical > 0 else 0.0

        return {
            "model_name": model_name,
            "sample_count": len(self.golden_data),
            "intent_metrics": {
                "accuracy": round(intent_acc * 100.0, 2),
                "macro_precision": round(float(prec) * 100.0, 2),
                "macro_recall": round(float(rec) * 100.0, 2),
                "macro_f1": round(float(f1) * 100.0, 2)
            },
            "triage_metrics": {
                "escalation_precision": round(float(t_prec) * 100.0, 2),
                "escalation_recall": round(float(t_rec) * 100.0, 2),
                "escalation_f1": round(float(t_f1) * 100.0, 2),
                "safety_recall_critical": round(safety_recall, 2),
                "dangerous_false_auto_rate": round(dangerous_false_auto_rate, 2),
                "total_critical_cases": total_critical,
                "missed_critical_count": dangerous_false_auto_count
            },
            "reply_quality": {
                "avg_judge_overall": round(float(np.mean(judge_overall_scores)), 2),
                "avg_judge_groundedness": round(float(np.mean(judge_groundedness)), 2),
                "avg_judge_brand_voice": round(float(np.mean(judge_brand_voice)), 2),
                "avg_judge_actionability": round(float(np.mean(judge_actionability)), 2),
                "avg_judge_escalation": round(float(np.mean(judge_escalation)), 2),
                "token_jaccard_similarity": round(float(np.mean(jaccard_scores)) * 100.0, 2),
                "pii_violation_count": pii_violations,
                "pii_violation_rate": round((pii_violations / len(self.golden_data)) * 100.0, 2)
            },
            "operational_latency": {
                "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "mean_latency_ms": round(float(np.mean(latencies)), 2)
            }
        }


def run_full_benchmark():
    evaluator = BenchmarkEvaluator()

    models = [
        ("Baseline 1 (Trivial Majority)", TrivialBaselineAgent()),
        ("Baseline 2 (Simple BM25)", SimpleBaselineAgent()),
        ("AmazonTrustAgent (Proposed)", AmazonTrustAgent())
    ]

    all_results = {}
    for name, model in models:
        print(f"Evaluating {name} on 200 golden test cases...")
        results = evaluator.evaluate_model(name, model)
        all_results[name] = results

    return all_results


if __name__ == "__main__":
    results = run_full_benchmark()
    print(json.dumps(results, indent=2))
