"""
RoadWatch AI demo app (Gradio).

    python app.py                 # opens http://127.0.0.1:7860
    python app.py --share         # temporary public link for judges

Needs weights/roadwatch_det.pt (and optionally weights/roadwatch_seg.pt).
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from roadwatch import DISCLAIMER
from roadwatch.detector import RoadWatch

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted(str(p) for p in (ROOT / "assets" / "examples").glob("*.jpg"))
rw: RoadWatch = None


def counts_md(counts, title):
    if not counts:
        return f"**{title}:** no road damage detected."
    rows = "\n".join(f"| {k.replace('_', ' ')} | {v} |" for k, v in sorted(counts.items()))
    return f"**{title}:** {sum(counts.values())} found\n\n| Hazard | Count |\n|---|---|\n{rows}"


def run_image(img_rgb, conf, use_seg, privacy):
    if img_rgb is None:
        return None, "Upload an image first.", "", []
    res = rw.analyze_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), conf, use_seg, privacy)
    table = [[d["hazard"].replace("_", " "), d["confidence"], str(d["box"])] for d in res["detections"]]
    return (cv2.cvtColor(res["image"], cv2.COLOR_BGR2RGB), res["quality"].as_markdown(),
            counts_md(res["counts"], "Detections"), table)


def run_video(video_path, conf, privacy, max_seconds, progress=gr.Progress()):
    if not video_path:
        return None, "Upload a video first."
    out = Path(tempfile.mkdtemp()) / "roadwatch_output.mp4"
    res = rw.analyze_video(video_path, out, conf, privacy, max_seconds or None,
                           progress=lambda f: progress(f, desc="Analysing frames"))
    q = res["quality_flagged"]
    note = (f"\n\n⚠️ {q:.0%} of frames flagged as low quality (blur, dark, glare or haze); "
            "results in those frames may be unreliable." if q > 0.2 else f"\n\n✅ Image quality OK in {1 - q:.0%} of frames.")
    return res["video"], counts_md(res["unique_counts"], f"Unique hazards in {res['frames']} frames "
                                   "(each tracked hazard counted once)") + note


def build():
    with gr.Blocks(title="RoadWatch AI") as demo:
        gr.Markdown("# 🛣️ RoadWatch AI: Road Surface & Hazard Detection\n"
                    "Detects **potholes, longitudinal cracks, transverse cracks and alligator cracks** "
                    "in road images and video, with an image-quality check and privacy blurring.")
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Road image", type="numpy")
                    conf = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Confidence threshold")
                    seg = gr.Checkbox(True, label="Show pothole & crack outlines (segmentation model)",
                                      interactive=rw.seg is not None)
                    priv = gr.Checkbox(True, label="Blur people and number plates")
                    btn = gr.Button("Detect road damage", variant="primary")
                with gr.Column():
                    out = gr.Image(label="Result")
                    qual = gr.Markdown()
                    summ = gr.Markdown()
            table = gr.Dataframe(headers=["Hazard", "Confidence", "Box [x1, y1, x2, y2]"], label="Detections")
            btn.click(run_image, [inp, conf, seg, priv], [out, qual, summ, table])
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp)
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    vin = gr.Video(label="Road video")
                    vconf = gr.Slider(0.05, 0.9, 0.3, step=0.05, label="Confidence threshold")
                    vpriv = gr.Checkbox(True, label="Blur people and number plates")
                    vmax = gr.Slider(0, 120, 20, step=5, label="Analyse first N seconds (0 = whole video)")
                    vbtn = gr.Button("Analyse video", variant="primary")
                with gr.Column():
                    vout = gr.Video(label="Result")
                    vsum = gr.Markdown()
            vbtn.click(run_video, [vin, vconf, vpriv, vmax], [vout, vsum])
        gr.Markdown(f"---\n*{DISCLAIMER}*")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(ROOT / "weights" / "roadwatch_det.pt"))
    ap.add_argument("--seg", default=str(ROOT / "weights" / "roadwatch_seg.pt"))
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu' for CPU (default: auto)")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    rw = RoadWatch(args.det, args.seg, device=args.device)
    build().launch(share=args.share)
