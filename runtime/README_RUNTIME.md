# Runtime Usage and Real-Time Pipeline

This file explains how to run the final inference system and how the real-time pipeline works internally.

## Main Runtime Command

Run the full real-time system from the project root:

```powershell
cd C:\GP\FallDetection_Final
python runtime\real_time_fall_detection_pipeline.py --source 0
```

`--source` can be:

- `0` for the default webcam
- another camera index such as `1`
- a video path
- an RTSP/HTTP stream URL

Save an annotated output video:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source "path\to\video.mp4" --output runtime_outputs\full_pipeline.mp4
```

Stop the live OpenCV window by pressing `Esc`.

## Models Used at Runtime

| Stage | File | Weight |
|---|---|---|
| Person detection/tracking | `patient_detection_tracking.py` | `model_weights/patient_detection_yolov8n.pt` |
| Role classification | `role_classifier.py` | `model_weights/role_classification_mobilenetv3_best.pt` |
| Pose extraction | `mediapipe_functions.py` | MediaPipe Pose |
| Fall classification | `ctrgcn_model.py` | `model_weights/fall_motion_ctrgcn_72f_impact.pt` |

## Real-Time Flow

For each incoming camera frame:

1. YOLO detects all people in the frame.
2. Ultralytics tracking assigns each person a persistent ID.
3. MobileNetV3 classifies each person crop as `patient`, `medical_staff`, or `other`.
4. The pipeline selects patient tracks for pose estimation.
5. MediaPipe extracts pose landmarks from the patient crop.
6. Landmarks are converted to 25 CTR-GCN joints.
7. Each patient track keeps a rolling 72-frame skeleton buffer.
8. CTR-GCN runs after the buffer is full and then repeats every `--infer-every` frames.
9. The frame overlay shows role, fall probability, motion status, and measured FPS.

## Important Timing Behavior

The final CTR-GCN model expects 72 frames.

At 18 FPS:

```text
72 frames / 18 FPS = 4 seconds
```

So the model observes roughly a 4-second motion window. The training clips were built as 2 seconds before impact and 2 seconds after impact.

In real time, the pipeline does not wait for separate clips. It continuously maintains a rolling buffer:

```text
frame 1 ... frame 72  -> first prediction
frame 7 ... frame 78  -> next prediction if --infer-every 6
frame 13 ... frame 84 -> next prediction
```

With a 30 FPS camera and `--infer-every 6`, the CTR-GCN prediction rate is approximately:

```text
30 / 6 = 5 predictions per second
```

The actual processing FPS depends on YOLO, MediaPipe, role classification, GPU/CPU speed, camera resolution, and number of tracked people.

## Alarm Logic

The runtime system fires the fall alarm only after **multiple consecutive CTR-GCN fall predictions** for the same tracked person.

By default:

```text
--fall-threshold 0.90
--infer-every 6
--alarm-confirmations 0
--alarm-confirmation-seconds 0.60
```

`--alarm-confirmations 0` means automatic mode. In automatic mode, the runtime calculates how many consecutive fall predictions are needed from the actual processing FPS of the running machine.

The decision is made separately for each tracked patient:

1. The patient track must have a full 72-frame skeleton buffer.
2. Every `--infer-every` frames, the latest 72-frame window is sent to CTR-GCN.
3. CTR-GCN outputs two probabilities: `non_fall` and `fall`.
4. The system reads the fall probability, `p_fall`.
5. If `p_fall >= --fall-threshold`, the track gets one fall confirmation.
6. If the next CTR-GCN prediction for the same track is also a fall, the confirmation counter increases again.
7. If any prediction is non-fall, the confirmation counter resets to zero.
8. The alarm fires only when the counter reaches the required confirmation count.

In code, the effective rule is:

```text
if alarm_confirmations > 0:
    required_confirmations = alarm_confirmations
else:
    prediction_rate = processing_fps / infer_every
    required_confirmations = round(prediction_rate * alarm_confirmation_seconds)
    required_confirmations = max(2, required_confirmations)

fall_prediction = p_fall >= fall_threshold

if fall_prediction:
    fall_streak += 1
else:
    fall_streak = 0

alarm = fall_streak >= required_confirmations
```

So with the default settings:

```text
alarm = about 0.60 seconds of consecutive CTR-GCN predictions with p_fall >= 0.90
```

The overlay shows the current confirmation counter as:

```text
fall p_fall=0.94 streak=2/3
```

With a machine processing 30 FPS and `--infer-every 6`, CTR-GCN runs about 5 times per second:

```text
30 FPS / 6 = 5 predictions per second
5 predictions/sec * 0.60 sec = 3 confirmations
```

With a slower machine processing 12 FPS and `--infer-every 6`, CTR-GCN runs about 2 times per second:

```text
12 FPS / 6 = 2 predictions per second
2 predictions/sec * 0.60 sec = about 1 confirmation
minimum automatic confirmation count = 2 confirmations
```

This keeps the alarm based on time instead of a fixed number of predictions, so slower machines do not wait too long and faster machines do not fire too easily.

Use a shorter confirmation time for a faster alarm:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --alarm-confirmation-seconds 0.40
```

Use a longer confirmation time for a more conservative alarm:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --alarm-confirmation-seconds 1.00
```

Force a fixed number of confirmations instead of automatic FPS-based mode:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --alarm-confirmations 3
```

## Useful Runtime Options

Run with the role classifier enabled:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0
```

Run without role classification and apply fall detection directly to tracked people:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --disable-role-classifier
```

Run fall detection on every tracked person even when the role classifier is enabled:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --include-non-patient
```

Use a stricter fall threshold:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --fall-threshold 0.95
```

Use a more conservative FPS-based alarm:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --alarm-confirmation-seconds 1.00
```

Force exactly 4 consecutive fall predictions before firing the alarm:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --alarm-confirmations 4
```

Run CTR-GCN more often:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --infer-every 3
```

Run CTR-GCN less often for better speed:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --infer-every 12
```

Headless run without display:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --no-display
```

## FPS Display

The live overlay shows:

- current frame number
- motion status
- fall probability
- measured processing FPS
- average processing FPS
- camera-reported FPS

Console FPS logs are printed every 5 seconds by default. Change this with:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --fps-log-every 10
```

Disable console FPS logs:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --fps-log-every 0
```

## Separate Runtime Scripts

Run only detection/tracking:

```powershell
python runtime\patient_detection_tracking.py --video "path\to\video.mp4"
```

Run detection/tracking without role classification:

```powershell
python runtime\patient_detection_tracking.py --video "path\to\video.mp4" --disable-role-classifier
```

Run only pose estimation and CTR-GCN motion classification with the 72-frame model:

```powershell
python runtime\pose_motion_classification.py --video "path\to\video.mp4"
```

## Recommended Real-Time Settings

For a single patient in a webcam scene:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --fall-threshold 0.90 --infer-every 6 --max-patients 1
```

For fewer false alarms:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --fall-threshold 0.95
```

For faster processing:

```powershell
python runtime\real_time_fall_detection_pipeline.py --source 0 --imgsz 640 --infer-every 12 --max-patients 1
```

## Output Folder

Runtime outputs should be written to:

```text
runtime_outputs/
```

This keeps generated videos and logs separate from code, weights, and evaluation results.
