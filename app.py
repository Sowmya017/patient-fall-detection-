"""
AI-Powered Fall Detection & Healthcare Monitoring Dashboard
Run with: streamlit run app.py
"""
import base64
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))

MODEL_DIR   = PROJECT_ROOT / "model_weights"
OUTPUT_DIR  = PROJECT_ROOT / "runtime_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

YOLO_WEIGHTS = str(MODEL_DIR / "patient_detection_yolov8n.pt")
ROLE_WEIGHTS = str(MODEL_DIR / "role_classification_mobilenetv3_best.pt")
FALL_WEIGHTS = str(MODEL_DIR / "fall_motion_ctrgcn_72f_impact.pt")
POSE_MODEL   = str(MODEL_DIR / "pose_landmarker_full.task")

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fall Detection Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e1e30,#2a2a44);
    border-radius:10px; padding:12px 16px;
    border-left:4px solid #4477ff;
}
/* Fall card accent */
.fall-metric { border-left-color:#ff4444 !important; }
.warn-metric { border-left-color:#ffaa44 !important; }
.safe-metric { border-left-color:#44cc88 !important; }

.event-card {
    background:#1e1e2e; border-radius:10px; padding:16px 20px;
    border-left:4px solid #ff4444; margin:8px 0;
}
.section-title { font-size:1.1rem; font-weight:600; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model loader  — models are loaded once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models (first run only)…")
def get_processor():
    from video_processor import VideoProcessor
    return VideoProcessor(
        yolo_weights=YOLO_WEIGHTS,
        role_weights=ROLE_WEIGHTS,
        fall_weights=FALL_WEIGHTS,
        pose_model=POSE_MODEL,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("Detection thresholds")

        # Sensitivity presets
        preset = st.radio("Sensitivity preset", ["Sensitive", "Balanced", "Conservative"],
                          index=1, horizontal=True,
                          help="Quick presets — override with sliders below")
        preset_defaults = {
            "Sensitive":    (0.65, 0.35),
            "Balanced":     (0.75, 0.45),
            "Conservative": (0.88, 0.55),
        }
        default_fall, default_warn = preset_defaults[preset]

        fall_thr = st.slider("Fall threshold",    0.40, 0.99, default_fall, 0.01,
                             help="Boosted probability above which a prediction counts as 'fall'")
        warn_thr = st.slider("Warning threshold", 0.25, 0.89, default_warn, 0.01,
                             help="Probability above which a yellow warning is shown")

        st.subheader("Classification")
        role_thr     = st.slider("Role confidence", 0.30, 0.90, 0.50, 0.05)
        disable_role = st.checkbox("Disable role classifier",
                                   help="Process all people, ignore patient/staff distinction")

        st.subheader("Performance")
        infer_every      = st.slider("Infer every N frames", 1, 15, 6, 1,
                                     help="CTR-GCN runs every N frames; higher = faster but coarser")
        role_every       = st.slider("Role refresh every N frames", 5, 30, 10, 5)
        max_pose_targets = st.slider("Max simultaneous pose targets", 1, 5, 3, 1)

        st.subheader("Alarm")
        alarm_secs = st.slider("Alarm confirmation (s)", 0.3, 2.0, 0.6, 0.1,
                               help="Seconds of consecutive fall predictions before alarm fires")

        st.divider()
        st.subheader("📊 Model performance")
        st.markdown("""
| Component | Detail |
|-----------|--------|
| Person detector | YOLOv8n |
| Role classifier | MobileNetV3-L |
| Pose estimator | MediaPipe Full |
| Fall detector | CTR-GCN 72-frame |
| Fall recall | **98.8 %** |
| Accuracy | **91.6 %** |
""")

    return dict(
        fall_threshold=fall_thr, warning_threshold=warn_thr,
        role_threshold=role_thr, disable_role_classifier=disable_role,
        infer_every=infer_every, role_every=role_every,
        max_pose_targets=max_pose_targets,
        alarm_confirmation_seconds=alarm_secs,
    )


# ---------------------------------------------------------------------------
# KPI row helper
# ---------------------------------------------------------------------------
def kpi_row(result):
    s = result.stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 People detected",  s["total_people"])
    c2.metric("🚨 Fall events",       s["total_falls"])
    c3.metric("⚠️ Warning cases",    s["warnings"])
    c4.metric("⚡ Processing FPS",   f"{s['processing_fps']:.1f}")
    c5.metric("🎞️ Frames processed", s["total_frames"])


# ---------------------------------------------------------------------------
# Tab 1 — Video Analysis
# ---------------------------------------------------------------------------
def tab_video(cfg: dict):
    st.header("📹 Video Analysis")

    col_up, col_info = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a video file", type=["mp4", "avi", "mov", "mkv", "wmv"],
            label_visibility="collapsed",
        )
    with col_info:
        st.info(f"""
**Active settings**
- Fall threshold: `{cfg['fall_threshold']}`
- Warning threshold: `{cfg['warning_threshold']}`
- Max pose targets: `{cfg['max_pose_targets']}`
- Infer every: `{cfg['infer_every']}` frames
        """)

    if uploaded is None:
        st.info("👆 Upload a video file to begin analysis.")
        return

    suffix     = Path(uploaded.name).suffix or ".mp4"
    input_path = str(OUTPUT_DIR / f"input_current{suffix}")
    output_path = str(OUTPUT_DIR / "output_current.mp4")

    # Save upload to disk
    with open(input_path, "wb") as f:
        f.write(uploaded.read())

    if st.button("▶️  Analyse Video", type="primary", use_container_width=True):
        with st.spinner("Warming up models…"):
            processor = get_processor()

        prog  = st.progress(0.0, text="Starting…")
        status = st.empty()

        def on_progress(fi, total):
            pct = fi / max(1, total)
            prog.progress(pct, text=f"Frame {fi} / {total}  ({pct*100:.0f} %)")

        try:
            result = processor.process_video(
                video_path=input_path,
                output_path=output_path,
                **cfg,
                progress_callback=on_progress,
            )
            st.session_state["result"] = result
            prog.progress(1.0, text="✅ Done!")
            status.empty()
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            raise

    # ---- Show results ----
    if "result" not in st.session_state:
        return

    result = st.session_state["result"]
    st.divider()
    kpi_row(result)

    st.divider()
    cv_col, pv_col = st.columns(2)
    with cv_col:
        st.subheader("Original")
        st.video(input_path)
    with pv_col:
        st.subheader("Processed  (skeletons + labels)")
        if os.path.exists(output_path):
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️  Download processed video", f.read(),
                    file_name="fall_detection_output.mp4", mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.warning("Processed video not found.")


# ---------------------------------------------------------------------------
# Tab 2 — Analytics
# ---------------------------------------------------------------------------
def tab_analytics():
    if "result" not in st.session_state:
        st.info("Process a video first to see analytics.")
        return

    result = st.session_state["result"]
    s = result.stats

    st.header("📊 Analytics Dashboard")
    kpi_row(result)
    st.divider()

    # Row 1: role pie + fall-prob timeline
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👔 Role distribution")
        role_counts = {
            "Patient":       s.get("patients", 0),
            "Medical staff": s.get("medical_staff", 0),
            "Other/Visitor": s.get("other", 0),
        }
        role_counts = {k: v for k, v in role_counts.items() if v > 0}
        if role_counts:
            fig = px.pie(
                values=list(role_counts.values()),
                names=list(role_counts.keys()),
                color_discrete_sequence=["#44cc88", "#4488ff", "#ffaa44"],
                hole=0.42,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                              height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No role data yet.")

    with col2:
        st.subheader("📈 Fall probability over time")
        if result.timeline:
            df = pd.DataFrame(result.timeline)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["time"], y=df["max_fall_prob"],
                fill="tozeroy", line_color="#ff5555", name="Max p(fall)",
            ))
            fig.add_hline(y=s.get("fall_threshold_used", 0.90),
                          line_dash="dash", line_color="orange",
                          annotation_text="Fall threshold", annotation_font_color="orange")
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Fall probability",
                yaxis_range=[0, 1], height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font_color="white", margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data.")

    # Row 2: people count + per-person table
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("👥 People in frame over time")
        if result.timeline:
            df = pd.DataFrame(result.timeline)
            fig = px.area(df, x="time", y="people",
                          color_discrete_sequence=["#4488ff"], labels={"people": "Count"})
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font_color="white", height=270, margin=dict(t=10, b=30),
                xaxis_title="Time (s)", yaxis_title="People",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("📋 Per-person summary")
        if result.track_summaries:
            rows = [
                dict(
                    ID=ts.track_id,
                    Role=ts.role.replace("_", " ").title(),
                    Confidence=f"{ts.role_conf:.0%}",
                    **{"Max p(fall)": f"{ts.max_fall_prob:.2f}"},
                    **{"Fall events": ts.num_fall_events},
                    **{"Warn frames": ts.warning_frames},
                    **{"Duration (s)": f"{ts.last_seen_time - ts.first_seen_time:.1f}"},
                )
                for ts in result.track_summaries.values()
            ]
            df_p = pd.DataFrame(rows)
            st.dataframe(df_p, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️  Download person report (CSV)",
                df_p.to_csv(index=False),
                file_name="person_report.csv", mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Tab 3 — Fall Events
# ---------------------------------------------------------------------------
def tab_events():
    if "result" not in st.session_state:
        st.info("Process a video first to see fall events.")
        return

    result = st.session_state["result"]
    st.header("⚠️ Fall Events")

    if not result.fall_events:
        st.success("✅ No fall events detected in this video.")
        return

    st.error(f"🚨 {len(result.fall_events)} fall event(s) detected")

    # Build table
    rows = [dict(**{
        "Event #":        i + 1,
        "Track ID":       ev.track_id,
        "Role":           ev.role.replace("_", " ").title(),
        "Start (s)":      f"{ev.start_time:.2f}",
        "End (s)":        f"{ev.end_time:.2f}",
        "Duration (s)":   f"{ev.duration:.2f}",
        "Peak p(fall)":   f"{ev.peak_probability:.3f}",
        "Start frame":    ev.start_frame,
    }) for i, ev in enumerate(result.fall_events)]
    df_ev = pd.DataFrame(rows)

    st.download_button(
        "⬇️  Download events report (CSV)",
        df_ev.to_csv(index=False),
        file_name="fall_events.csv", mime="text/csv",
    )
    st.dataframe(df_ev, use_container_width=True, hide_index=True)
    st.divider()

    # Gantt-style timeline
    st.subheader("📅 Events timeline")
    if result.fall_events:
        fig = go.Figure()
        for i, ev in enumerate(result.fall_events):
            x0 = ev.start_time
            x1 = max(ev.end_time, ev.start_time + 0.3)
            fig.add_shape(type="rect",
                          x0=x0, x1=x1, y0=i - 0.4, y1=i + 0.4,
                          fillcolor="#cc3333", opacity=0.75, line_color="#ff0000")
            fig.add_annotation(x=(x0+x1)/2, y=i,
                               text=f"ID {ev.track_id}  {ev.peak_probability:.0%}",
                               showarrow=False, font=dict(color="white", size=11))
        fig.update_layout(
            xaxis=dict(title="Time (s)", range=[0, result.duration]),
            yaxis=dict(title="", tickmode="array",
                       tickvals=list(range(len(result.fall_events))),
                       ticktext=[f"Event #{i+1}" for i in range(len(result.fall_events))]),
            height=max(180, len(result.fall_events) * 60 + 80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
            font_color="white", margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Event cards with snapshots
    st.subheader("🖼️ Event snapshots")
    for i, ev in enumerate(result.fall_events):
        with st.expander(
            f"🚨 Event #{i+1} — Track {ev.track_id}  "
            f"@ {ev.start_time:.1f}s  |  peak p(fall)={ev.peak_probability:.1%}",
            expanded=(i == 0),
        ):
            sc, ic = st.columns([2, 3])
            with sc:
                if ev.snapshot_b64:
                    st.image(
                        base64.b64decode(ev.snapshot_b64),
                        caption=f"Peak frame  p={ev.peak_probability:.2f}",
                        use_container_width=True,
                    )
                else:
                    st.info("No snapshot captured for this event.")
            with ic:
                st.markdown(f"""
| | |
|--|--|
| **Track ID** | {ev.track_id} |
| **Role** | {ev.role.replace('_',' ').title()} |
| **Start** | {ev.start_time:.2f} s (frame {ev.start_frame}) |
| **End** | {ev.end_time:.2f} s (frame {ev.end_frame}) |
| **Duration** | {ev.duration:.2f} s |
| **Peak p(fall)** | {ev.peak_probability:.1%} |
""")


# ---------------------------------------------------------------------------
# Tab 4 — Live Monitor
# ---------------------------------------------------------------------------
def tab_live():
    st.header("📡 Live Monitoring")

    col_cfg, col_cmd = st.columns([1, 2])
    with col_cfg:
        st.subheader("Settings")
        source     = st.text_input("Camera source", "0",
                                   help="Index (0,1,…), RTSP URL, or video path")
        live_thr   = st.slider("Fall threshold", 0.50, 0.99, 0.90, 0.01, key="live_thr")
        max_p      = st.slider("Max pose targets", 1, 5, 1, key="live_max")
        no_display = st.checkbox("Headless (no window)", value=False)

    with col_cmd:
        st.subheader("Launch command")
        flags = f"--source {source} --fall-threshold {live_thr} --max-patients {max_p}"
        if no_display:
            flags += " --no-display"
        cmd = (
            f"python runtime/real_time_fall_detection_pipeline.py \\\n"
            f"    {flags} \\\n"
            f"    --yolo-weights model_weights/patient_detection_yolov8n.pt \\\n"
            f"    --role-weights model_weights/role_classification_mobilenetv3_best.pt \\\n"
            f"    --fall-weights model_weights/fall_motion_ctrgcn_72f_impact.pt"
        )
        st.code(cmd, language="bash")
        st.info("Run the command above in a terminal to open the live detection window.  "
                "Press **ESC** inside that window to stop.")

    st.divider()
    st.subheader("📸 Snapshot test  (single frame)")
    snap = st.camera_input("Capture a frame from your webcam")
    if snap:
        st.image(snap, caption="Captured frame", use_container_width=True)
        st.info("Single-frame preview only.  "
                "Full real-time fall detection requires the terminal pipeline above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("🏥 AI Fall Detection & Healthcare Monitoring")
    st.caption("YOLOv8 · MobileNetV3 · MediaPipe Pose · CTR-GCN")

    cfg = render_sidebar()

    t1, t2, t3, t4 = st.tabs([
        "📹 Video Analysis",
        "📊 Analytics",
        "⚠️ Fall Events",
        "📡 Live Monitor",
    ])

    with t1: tab_video(cfg)
    with t2: tab_analytics()
    with t3: tab_events()
    with t4: tab_live()


if __name__ == "__main__":
    main()
