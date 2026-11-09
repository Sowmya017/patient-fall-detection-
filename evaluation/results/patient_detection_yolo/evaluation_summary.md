# Patient Detection YOLO Evaluation Summary

The runtime detector uses the YOLOv8n person class (`class_id = 0`) with ByteTrack-style tracking through Ultralytics.

Raw labeled detection datasets are not included in this final package. The included evaluation script `evaluation/scripts/evaluate_patient_detection_yolo.py` evaluates a video by reporting:

- total frames
- frames with no person detection
- no-detection rate
- mean and median person count per frame
- mean and median detection confidence
- optional annotated output video

This keeps the final repository dataset-free while preserving the evaluation method used for detector sanity checks.
