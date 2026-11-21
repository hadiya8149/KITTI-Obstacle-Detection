"""Gradio demo: upload a road image -> detected obstacles, distance estimate, warning level and speed.
Written by notebooks/08_speed_and_app.ipynb (section 5).

    python app/app.py            # opens http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import cv2
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from pipeline import ObstacleDetector, WEIGHTS_DIR  # noqa: E402

MODELS = {"FINAL (default)": "FINAL", "E6 (weather-augmented)": "E6"}
_DETECTORS = {}
EXAMPLES = sorted(str(p) for p in (ROOT / "app" / "examples").glob("*.png"))


def detector(name):
    if name not in _DETECTORS:
        _DETECTORS[name] = ObstacleDetector(WEIGHTS_DIR / f"{name}.pt")
    return _DETECTORS[name]


def run(image, conf, model="FINAL (default)"):
    if image is None:
        return None, "Upload an image first.", None
    out = detector(MODELS.get(model, model)).predict(image, conf=conf)
    rgb = cv2.cvtColor(out["image"], cv2.COLOR_BGR2RGB)
    table = out["detections"][["type", "score", "distance_m", "in_corridor", "risk"]] \
        .sort_values("distance_m").rename(columns={"type": "class", "distance_m": "distance (m, approx.)"})
    status = f"### {out['level']}\n{out['message']}\n\nInference: {out['ms']:.0f} ms ({1000 / out['ms']:.0f} FPS)"
    return rgb, status, table


with gr.Blocks(title="KITTI obstacle detection") as demo:
    gr.Markdown("# Camera-based obstacle detection\nYOLOv8s fine-tuned on KITTI (Group 16). Upload a front-camera "
                "road image. Distances are rough pinhole estimates that assume a KITTI-like camera.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Road image", sources=["upload"])
            model = gr.Radio(list(MODELS), value="FINAL (default)", label="Model")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Detections")
            status = gr.Markdown()
    table = gr.Dataframe(label="Obstacles (nearest first)")
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=inp)
    btn.click(run, [inp, conf, model], [out_img, status, table])
    inp.change(run, [inp, conf, model], [out_img, status, table])

if __name__ == "__main__":
    demo.launch()
