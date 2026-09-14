"""Grounded natural-language explanation generation.

Explanations are assembled from *measured* quantities only: the calibrated
probability, the Grad-CAM regions, and the concentration of attention.  No
external LLM and no hallucinated evidence.  All wording communicates
uncertainty honestly (never an absolute accusation).
"""

from __future__ import annotations

from typing import Dict, List

VERDICT_LABELS = {
    "likely_ai_generated": "Likely AI-generated",
    "likely_real": "Likely real",
    "inconclusive": "Inconclusive",
}

EVIDENCE_TEMPLATES = [
    "The model's strongest response is concentrated around {region}.",
    "Strong model activation is localized around {region}.",
    "Secondary activation appears around {region}.",
]


def build_verdict(probability_ai: float, high: float, low: float) -> Dict[str, str]:
    """Map a calibrated probability to a responsible verdict."""
    if probability_ai >= high:
        verdict, status = "likely_ai_generated", "high_confidence"
    elif probability_ai <= low:
        verdict, status = "likely_real", "high_confidence"
    else:
        verdict, status = "inconclusive", "low_confidence"
    return {"verdict": verdict, "status": status, "label": VERDICT_LABELS[verdict]}


def _format_confidence(probability_ai: float, verdict: str) -> float:
    """Confidence = probability of the dominant class for decisive verdicts;
    for inconclusive predictions we report the raw probability."""
    if verdict == "likely_ai_generated":
        return probability_ai
    if verdict == "likely_real":
        return 1.0 - probability_ai
    return max(probability_ai, 1.0 - probability_ai)


def build_explanation(
    probability_ai: float,
    regions: List[Dict[str, object]],
    concentration: Dict[str, float],
    image_shape,
    verdict: str,
    high_threshold: float,
    low_threshold: float,
) -> Dict[str, object]:
    """Assemble the explanation object for a prediction.

    Returns:
        summary (str), evidence (list of str), uncertainty (str).
    """
    confidence = _format_confidence(probability_ai, verdict)
    pct = int(round(confidence * 100))

    # ---- evidence sentences (grounded in Grad-CAM regions) ----
    from src.explainability.evidence import region_position_label

    evidence: List[str] = []
    if regions:
        first = region_position_label(regions[0]["bbox"], image_shape)
        evidence.append(
            f"The model's strongest response is concentrated around {first}."
        )
        if len(regions) >= 2:
            second = region_position_label(regions[1]["bbox"], image_shape)
            evidence.append(f"Additional activation appears around {second}.")
        if len(regions) >= 3:
            third = region_position_label(regions[2]["bbox"], image_shape)
            evidence.append(f"A third, weaker region of activation is near {third}.")
    else:
        evidence.append(
            "The model's attention is diffuse across the image rather than "
            "localized to a single region."
        )

    if concentration.get("top5pct_concentration", 0) < 1.5:
        evidence.append(
            "Because the attention is spread out, this explanation is weaker "
            "than a strongly localized one."
        )

    # ---- summary + uncertainty ----
    if verdict == "likely_ai_generated":
        summary = (
            f"Likely AI-generated — {pct}% confidence. "
            "The model estimates a high likelihood that this image was synthetically generated."
        )
    elif verdict == "likely_real":
        summary = (
            f"Likely real — {pct}% confidence. "
            "The model estimates a high likelihood that this is a real photograph."
        )
    else:
        summary = (
            f"Inconclusive — {pct}% confidence. "
            "The model does not have enough evidence to make a strong prediction."
        )

    uncertainty = (
        "This is a likelihood assessment based on learned visual patterns, "
        "not a definitive determination. The highlighted regions show where the "
        "model looked; they are not proof of any specific artifact."
    )

    return {
        "summary": summary,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "confidence": confidence,
    }
