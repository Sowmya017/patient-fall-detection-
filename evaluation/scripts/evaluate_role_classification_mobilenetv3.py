import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from role_classifier import MobileNetRoleClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MobileNetV3 role classifier on an ImageFolder dataset.")
    parser.add_argument("--data-dir", required=True, help="ImageFolder root with one class folder per role.")
    parser.add_argument("--weights", default="model_weights/role_classification_mobilenetv3_best.pt")
    parser.add_argument("--out-dir", default="evaluation/results/role_classification_mobilenetv3")
    return parser.parse_args()


def draw_confusion_matrix(matrix, labels, output_path, normalize=False):
    values = matrix.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.maximum(row_sums, 1), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else str(int(values[i, j]))
            ax.text(j, i, text, ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = MobileNetRoleClassifier(args.weights)
    dataset = ImageFolder(args.data_dir)
    class_names = dataset.classes

    rows = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for image_path, true_index in dataset.samples:
            image = Image.open(image_path).convert("RGB")
            tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
            logits = classifier.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_index = int(np.argmax(probabilities))
            y_true.append(true_index)
            y_pred.append(pred_index)
            rows.append(
                {
                    "image_path": image_path,
                    "true_label": class_names[true_index],
                    "pred_label": classifier.idx_to_class[pred_index],
                    "confidence": float(probabilities[pred_index]),
                }
            )

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(rows).to_csv(out_dir / "per_sample_results.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "weights": str(args.weights),
                "num_samples": len(rows),
                "classes": class_names,
                "confusion_matrix": matrix.astype(int).tolist(),
                "classification_report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_confusion_matrix(matrix, class_names, out_dir / "confusion_matrix.png")
    draw_confusion_matrix(matrix, class_names, out_dir / "confusion_matrix_normalized.png", normalize=True)
    print(f"Saved role-classification evaluation to {out_dir}")


if __name__ == "__main__":
    main()
