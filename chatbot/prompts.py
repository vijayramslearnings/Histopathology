"""System prompt definitions and deterministic fallback intent engine for histopathology AI chatbot."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


SYSTEM_PROMPT = """You are an explanation assistant for an experimental histopathology AI model.

You must explain only the information provided in the model result.

Never invent model findings.

Never provide a medical diagnosis.

Never claim that a highlighted region proves cancer.

Clearly distinguish model prediction from clinical interpretation.

If information is unavailable, explicitly say that it is unavailable.

Use simple language unless the user asks for technical details."""


class DeterministicIntentEngine:
    """Fallback intent matcher that generates factual responses directly from model JSON findings."""

    @staticmethod
    def answer(user_message: str, findings: Dict[str, Any]) -> str:
        msg = user_message.strip().lower()
        
        # Extract fields from findings
        pred = findings.get("prediction", {})
        cancer_type = pred.get("cancer_type", "Unknown")
        confidence = pred.get("confidence", 0.0)
        conf_pct = round(confidence * 100.0, 1)
        class_probs = pred.get("class_probabilities", {})

        subtype = findings.get("subtype", {})
        subtype_label = subtype.get("label")
        subtype_conf = subtype.get("confidence")

        scale_att = findings.get("scale_attention", {})
        dom_scale = findings.get("dominant_scale", "unknown")

        regions = findings.get("important_regions", [])
        faithfulness = findings.get("faithfulness", {})
        stability = findings.get("stability", {})
        uncertainty = findings.get("uncertainty", {})

        organ_type = findings.get("organ_type", "Auto")
        has_cancer = findings.get("has_cancer", True)
        cancer_status = findings.get("cancer_status", f"Cancer Detected ({cancer_type})")

        # Organ context question
        if "organ" in msg or "tissue" in msg:
            return f"The specified/analyzed organ origin for this image is {organ_type}."

        # Medical diagnosis / proof questions
        if any(term in msg for term in ["prove", "diagnosis", "do i have cancer", "patient have cancer", "is it malignant", "am i sick"]):
            return (
                f"The model predicted {cancer_type} for {organ_type} tissue with {conf_pct}% confidence, but this research model cannot establish a medical diagnosis. "
                "The highlighted regions represent areas that influenced the model's prediction."
            )

        # Cancer presence question ("is it cancer?", "does it have cancer?", "has cancer")
        if any(term in msg for term in ["have cancer", "is it cancer", "has cancer", "is cancer", "cancer detected"]):
            status_text = f"Cancer Detected ({cancer_type})" if has_cancer else "Non-Cancerous / Benign"
            return (
                f"For this {organ_type} image, the model assessment is: {status_text} with {conf_pct}% confidence. "
                "These results represent experimental model outputs and are not a clinical diagnosis."
            )

        # 1. "What did the model predict?"
        if any(k in msg for k in ["what did the model predict", "what is the prediction", "what did model find", "prediction result"]):
            res = f"The model predicted {cancer_type} with {conf_pct}% confidence."
            if subtype_label:
                res += f" Subtype prediction: {subtype_label} ({round(subtype_conf*100, 1) if subtype_conf else 0}% confidence)."
            return res

        # 2. "How confident is the prediction?" / "Why is confidence low?"
        if "confidence" in msg or "how confident" in msg:
            if uncertainty.get("is_low_confidence"):
                return (
                    f"The model predicted {cancer_type} with {conf_pct}% confidence, which is below the high-confidence threshold. "
                    "Low-confidence model prediction. Review the image and model explanation carefully."
                )
            return f"The model's confidence for {cancer_type} is {conf_pct}%. Top-2 confidence margin is {round(uncertainty.get('top2_confidence', 0)*100, 1)}%."

        # 3. "Why did the model make this prediction?" / "Why?"
        if msg in ["why", "why?", "why this prediction", "why did it predict", "why predicted"] or "why did the model" in msg:
            res = (
                f"The model assigned the highest probability ({conf_pct}%) to {cancer_type}. "
                f"Its explanation indicates that the prediction was influenced by the learned scale attention (dominant scale: {dom_scale}) "
                "and patch attention regions. These visual explanations describe model behavior rather than providing a clinical interpretation."
            )
            return res

        # 4. "Which regions were important?" / "Which regions received highest attention?" / "Which region?"
        if any(k in msg for k in ["which region", "important region", "highest attention", "top regions"]):
            if not regions:
                return "That information is not available from the current model output."
            top_r = regions[0]
            score = top_r.get("attention_score", 0.0)
            coord = top_r.get("coordinates", {})
            return (
                f"Region #1 at scale {top_r.get('scale', 'high-scale')} (x={coord.get('x')}, y={coord.get('y')}) "
                f"received the highest attention score of {score:.2f}. A total of {len(regions)} top regions were extracted."
            )

        # 5. "Which scale was most important?" / "What scale did model rely on?"
        if "scale" in msg:
            if not scale_att:
                return "That information is not available from the current model output."
            scale_breakdown = ", ".join([f"{k}: {round(v*100, 1)}%" for k, v in scale_att.items()])
            return (
                f"The model assigned the highest scale attention to the {dom_scale} representation. "
                f"Scale breakdown: {scale_breakdown}."
            )

        # 6. "What does the heatmap show?" / "What does Grad-CAM mean?"
        if "heatmap" in msg or "gradcam" in msg or "grad-cam" in msg:
            return (
                "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights image regions that contributed "
                f"most strongly to the model's prediction of {cancer_type}. Brighter red/yellow zones indicate areas of higher model activation."
            )

        # 7. "What is the predicted subtype?"
        if "subtype" in msg:
            if subtype_label:
                return f"The predicted subtype is {subtype_label} with {round(subtype_conf*100, 1) if subtype_conf else 0}% confidence."
            return subtype.get("message", "Subtype prediction is not available for this sample.")

        # 8. "How reliable is this explanation?" / "What does stability score mean?"
        if "stability" in msg or "reliable" in msg:
            s_val = stability.get("ssim", 0.0)
            ps_val = stability.get("prediction_stability", 1.0)
            return (
                f"The explanation stability evaluation yields a Structural Similarity (SSIM) score of {s_val:.2f} "
                f"and prediction stability of {round(ps_val*100, 1)}% under rotation, brightness, contrast, and crop perturbations."
            )

        # 9. "What happened during the deletion test?" / "faithfulness"
        if "deletion" in msg or "insertion" in msg or "faithful" in msg:
            del_auc = faithfulness.get("deletion_auc")
            ins_auc = faithfulness.get("insertion_auc")
            if del_auc is not None and ins_auc is not None:
                return (
                    f"During the faithfulness evaluation, the Deletion AUC was {del_auc:.3f} and Insertion AUC was {ins_auc:.3f}. "
                    "A steep decrease in confidence during patch deletion indicates that the highlighted regions were influential in the model's prediction."
                )
            return "That information is not available from the current model output."

        # 10. "What are the alternative predictions?" / "What about other classes?"
        if any(k in msg for k in ["other classes", "alternative", "other predictions", "probabilities"]):
            probs_formatted = ", ".join([f"{k}: {round(v*100, 1)}%" for k, v in class_probs.items()])
            return f"The model evaluated the following class probabilities: {probs_formatted}."

        # 11. Patch attention explanation
        if "patch attention" in msg:
            return (
                "Patch attention measures the relative importance assigned by the Multiple Instance Learning (MIL) module "
                "to individual spatial patches across image scales."
            )

        # Default factual fallback
        return (
            f"The model predicted {cancer_type} with {conf_pct}% confidence. "
            f"Dominant scale attention is {dom_scale}. "
            "These outputs describe experimental AI model behavior and are not a clinical diagnosis."
        )
