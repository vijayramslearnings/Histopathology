"""Scale attention score processor and multi-scale contribution analyzer."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def process_scale_attention(
    scale_weights: List[float] | tuple[float, ...] | Any,
    scale_names: tuple[str, ...] = ("5x", "10x", "20x"),
    use_context_labels: bool = False,
) -> Dict[str, Any]:
    """Process and normalize scale attention weights across multi-scale branches."""
    weights = list(scale_weights)
    if len(weights) != len(scale_names):
        # Adjust default scale names if length mismatch
        scale_names = tuple(f"scale_{i+1}" for i in range(len(weights)))

    if use_context_labels:
        label_mapping = {
            "5x": "low-scale",
            "10x": "medium-scale",
            "20x": "high-scale",
        }
        effective_names = [label_mapping.get(name, name) for name in scale_names]
    else:
        effective_names = list(scale_names)

    # Normalize weights so they sum strictly to 1.0
    total = sum(weights)
    if total > 0:
        normalized = [round(w / total, 4) for w in weights]
    else:
        normalized = [round(1.0 / len(weights), 4) for _ in weights]

    scale_dict: Dict[str, float] = {
        name: norm for name, norm in zip(effective_names, normalized)
    }

    # Identify dominant scale
    dominant_scale = max(scale_dict.items(), key=lambda item: item[1])[0]
    dominant_pct = round(scale_dict[dominant_scale] * 100.0, 1)

    summary_text = (
        f"The model assigned the highest learned importance to the {dominant_scale} representation ({dominant_pct}%)."
    )

    return {
        "scale_attention": scale_dict,
        "dominant_scale": dominant_scale,
        "dominant_percentage": dominant_pct,
        "summary": summary_text,
    }
