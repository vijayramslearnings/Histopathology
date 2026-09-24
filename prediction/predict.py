"""CLI for research-only single-image prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.preprocessing import BaselinePreprocessor, build_preprocessor
from xai.attention_maps import attention_heatmap, save_attention_heatmap
from xai.gradcam import GradCAM, save_gradcam_overlay

from .inference import load_checkpoint, predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to one RGB/RGBA image.")
    parser.add_argument("--checkpoint", required=True, help="Saved .pt checkpoint.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--gradcam",
        help="Optional output path for a Grad-CAM overlay of the predicted cancer class.",
    )
    parser.add_argument(
        "--attention-heatmap",
        help="Optional output path for the proposed-model patch attention heatmap.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    result = predict_image(
        model=model,
        image_path=args.image,
        cancer_classes=checkpoint["cancer_classes"],
        subtype_classes=checkpoint.get("subtype_classes", []),
        image_size=checkpoint.get("image_size", 224),
        model_name=checkpoint["model_config"]["name"],
        device=args.device,
            scale_fractions=tuple(checkpoint.get("scale_fractions", (1.0, 0.75, 0.55))),
    )
    serialized = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if args.gradcam:
        from PIL import Image

        preprocessor = build_preprocessor(
            checkpoint.get("image_size", 224),
            train=False,
            scale_fractions=tuple(
                checkpoint.get("scale_fractions", (1.0, 0.75, 0.55))
            ),
        )
        with Image.open(args.image) as image:
            model_input = preprocessor(image).unsqueeze(0).to(args.device)
        if checkpoint["model_config"]["name"] == "baseline":
            baseline_preprocessor = BaselinePreprocessor(preprocessor.config)
            with Image.open(args.image) as image:
                model_input = baseline_preprocessor(image).unsqueeze(0).to(args.device)
        cam = GradCAM(model, model.target_layer())
        try:
            heatmap = cam(
                model_input,
                class_index=checkpoint["cancer_classes"].index(result["cancer_type"]),
            )
        finally:
            cam.close()
        save_gradcam_overlay(args.image, heatmap, args.gradcam)

    if args.attention_heatmap:
        if checkpoint["model_config"]["name"] != "multiscale":
            raise ValueError("Attention heatmaps require a multiscale checkpoint.")
        heatmap = attention_heatmap(result["patch_weights"])
        save_attention_heatmap(heatmap, args.attention_heatmap)

    print(serialized)


if __name__ == "__main__":
    main()