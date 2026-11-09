# 72-Frame Sliding-Window Fall Evaluation

## Setup

- Weights: `C:\GP\CTRGCN\CTR-GCN\work_dir\harup_mediapipe25joints_72f_impact_v2\runs-20-640.pt`
- Non-fall roots: `F:\GP_Dataset\activities2; F:\GP_Dataset\activities6-11`
- Fall roots: `F:\GP_Dataset\Subjects`
- Window size: 72 frames
- Window stride: 18 frames
- Aggregation: video/sample is fall if any window has fall probability >= 0.50

## Metrics

- Evaluated samples: 655
- Valid samples: 654
- No-pose samples: 1
- Accuracy: 0.9159
- Fall precision: 0.9122
- Fall recall: 0.9883
- Fall F1: 0.9487
- Macro F1: 0.8573
- ROC AUC: 0.959768107843822

## Confusion Matrix

| True \ Predicted | non_fall | fall |
|---|---:|---:|
| non_fall | 90 | 49 |
| fall | 6 | 509 |

## Files

- `per_sample_results.csv`
- `metrics.json`

## Threshold Sweep

| Threshold | Accuracy | Fall Precision | Fall Recall | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.9067 | 0.8969 | 0.9961 | 59 | 2 |
| 0.20 | 0.9083 | 0.9012 | 0.9922 | 56 | 4 |
| 0.30 | 0.9128 | 0.9089 | 0.9883 | 51 | 6 |
| 0.40 | 0.9144 | 0.9106 | 0.9883 | 50 | 6 |
| 0.50 | 0.9159 | 0.9122 | 0.9883 | 49 | 6 |
| 0.60 | 0.9190 | 0.9170 | 0.9864 | 46 | 7 |
| 0.70 | 0.9220 | 0.9203 | 0.9864 | 44 | 7 |
| 0.80 | 0.9190 | 0.9200 | 0.9825 | 44 | 9 |
| 0.85 | 0.9205 | 0.9232 | 0.9806 | 42 | 10 |
| 0.90 | 0.9266 | 0.9300 | 0.9806 | 38 | 10 |
| 0.95 | 0.9327 | 0.9369 | 0.9806 | 34 | 10 |
| 0.97 | 0.9312 | 0.9417 | 0.9728 | 31 | 14 |
| 0.99 | 0.9327 | 0.9486 | 0.9670 | 27 | 17 |

## Error Files

- False positives at threshold 0.50: 49 (`false_positives.csv`)
- False negatives at threshold 0.50: 6 (`false_negatives.csv`)