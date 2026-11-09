# MobileNetV3 Role Classification Evaluation - Run 3

This folder contains the final evaluation artifacts for the role-classification model used in the patient fall detection runtime pipeline.

## Purpose

The role classifier receives person crops from the YOLO tracker and classifies each crop into one of three runtime roles:

- `medical_staff`
- `other`
- `patient`

The real-time fall pipeline uses this prediction to decide which tracked person should be passed to MediaPipe pose extraction and CTR-GCN fall classification.

## Source Run

- Run name: `Run_3`
- Source folder: `C:\Users\Victus\Downloads\Run_3-20260603T143947Z-3-001\Run_3`
- Original training output: `/content/drive/MyDrive/GP Files/HIOD/runs/Run_3`
- Model architecture: `mobilenet_v3_large`
- Number of classes: `3`

## Dataset Mapping

The training configuration maps source labels into runtime labels as follows:

| Source label | Runtime label |
|---|---|
| `staff` | `medical_staff` |
| `patient` | `patient` |
| `visitor` | `other` |
| `person` | `other` |

## Preprocessing

Person crops were generated from the hospital-scene dataset before training:

- Crop padding: `0.15`
- Minimum box area: `2500`
- Maximum aspect ratio: `6.0`
- Deduplication: enabled
- Perceptual-hash dedup threshold: `6`
- Debug samples saved: `50`

## Training Setup

| Setting | Value |
|---|---:|
| Model | `mobilenet_v3_large` |
| Pretrained backbone | `true` |
| Image size | `224` |
| Batch size | `16` |
| Configured epochs | `45` |
| Completed epochs in history | `29` |
| Initial learning rate | `0.0003` |
| Minimum learning rate | `0.00001` |
| Weight decay | `0.0001` |
| Label smoothing | `0.03` |
| AMP | `true` |
| Class weights | `true` |
| Weighted sampler | `true` |
| Early stopping patience | `10` |

Augmentation:

- Horizontal flip: `0.5`
- Color jitter: `0.25`
- Random erasing: `0.15`
- Random crop scale: `[0.8, 1.0]`
- Rotation: `10` degrees
- Translate: `0.05`

## Selected Model

The selected role-classification model is the **best checkpoint from epoch 19**, chosen by validation macro F1.

- Best checkpoint: `/content/drive/MyDrive/GP Files/HIOD/runs/Run_3/best.pt`
- Last checkpoint: `/content/drive/MyDrive/GP Files/HIOD/runs/Run_3/last.pt`
- Best epoch: `19`
- Best macro F1: `0.7785515630313449`

Although epoch 25 reached the highest validation accuracy and epoch 29 reached the highest weighted F1, epoch 19 was selected because macro F1 better reflects balanced performance across `medical_staff`, `other`, and `patient`.

## Training History Highlights

| Metric | Epoch | Value |
|---|---:|---:|
| Best validation accuracy | 25 | `0.8092592593` |
| Best validation macro F1 | 19 | `0.7785515630` |
| Best validation weighted F1 | 29 | `0.8136342804` |
| Last epoch validation accuracy | 29 | `0.8092592593` |
| Last epoch validation macro F1 | 29 | `0.7743327739` |
| Last epoch validation weighted F1 | 29 | `0.8136342804` |

Total recorded training time: `761.9968` seconds.

## Final Validation Report

```text
=== Final validation report (BEST checkpoint) ===
               precision    recall  f1-score   support

medical_staff       0.89      0.81      0.85       340
        other       0.59      0.70      0.64        97
      patient       0.81      0.89      0.85       103

     accuracy                           0.81       540
    macro avg       0.76      0.80      0.78       540
 weighted avg       0.82      0.81      0.81       540
```

## Confusion Matrices

Original Run 3 output paths:

```text
/content/drive/MyDrive/GP Files/HIOD/runs/Run_3/confusion_matrix.png
/content/drive/MyDrive/GP Files/HIOD/runs/Run_3/confusion_matrix_normalized.png
```

Final package paths:

- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

## Files in This Folder

| File | Description |
|---|---|
| `README.md` | Single human-readable summary of the Run 3 role-classification evaluation |
| `metrics.json` | Structured evaluation summary and selected-checkpoint metadata |
| `training_history.json` | Per-epoch training loss, validation accuracy, macro F1, weighted F1, LR, and epoch time |
| `config_used.yaml` | Training, preprocessing, data mapping, augmentation, and evaluation configuration |
| `confusion_matrix.png` | Final confusion matrix for the selected best checkpoint |
| `confusion_matrix_normalized.png` | Normalized confusion matrix for the selected best checkpoint |

## Runtime Weight

The runtime pipeline uses the packaged MobileNetV3 role-classifier checkpoint:

```text
model_weights/role_classification_mobilenetv3_best.pt
```
