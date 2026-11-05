"""
Hand Gesture Recognition demo (Peace / Stop / Thumbs Up)
Assignment 2, Computer Vision, Astana IT University
Author: Berdimbet Alima (ST2405)

Model: frozen DINOv2-small features + logistic regression (trained with synthetic conditions).
Features: upload or webcam, synthetic stress-test generator, probabilities,
occlusion-sensitivity explanation, latency and device information.

Run:
    pip install -r requirements.txt
    python app.py
Files needed next to this script:
    dinov2_logreg_synth.joblib   (trained classifier)
    class_names.json             (["peace", "stop", "thumbs_up"])
The DINOv2 weights are downloaded automatically from Hugging Face.
"""
import json
import os
import time

import joblib
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import AutoModel

try:  # optional: lets PIL open iPhone HEIC photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(HERE, "dinov2_logreg_synth.joblib")
CLASSES_PATH = os.path.join(HERE, "class_names.json")
DINO_NAME = "facebook/dinov2-small"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
RNG = np.random.default_rng(42)

# Filled in by load_models()
STATE = {"dino": None, "clf": None, "classes": None, "device": "cpu", "device_name": "CPU"}


# ----------------------------------------------------------------------------
# Synthetic stress-test generator (these operations produce SYNTHETIC images)
# ----------------------------------------------------------------------------
def motion_blur(img, k):
    arr = np.asarray(img).astype(np.float32)
    axis = int(RNG.integers(0, 2))
    acc = sum(np.roll(arr, s - k // 2, axis=axis) for s in range(k)) / k
    return Image.fromarray(acc.clip(0, 255).astype(np.uint8))


def add_noise(img, sigma):
    arr = np.asarray(img).astype(np.float32)
    arr = arr + RNG.normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def apply_condition(img, cond, strength):
    img = img.convert("RGB").resize((224, 224))
    if cond == "Dark":
        return ImageEnhance.Brightness(img).enhance(1 - 0.8 * strength)
    if cond == "Backlight":
        img = ImageEnhance.Brightness(img).enhance(1 + 1.0 * strength)
        return ImageEnhance.Contrast(img).enhance(1 - 0.5 * strength)
    if cond == "Motion blur":
        return motion_blur(img, int(3 + 20 * strength))
    if cond == "Noise":
        return add_noise(img, 50 * strength)
    return img


# ----------------------------------------------------------------------------
# Model: DINOv2 features + logistic regression
# ----------------------------------------------------------------------------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["device"] = device
    STATE["device_name"] = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    STATE["dino"] = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    STATE["clf"] = joblib.load(CLASSIFIER_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        STATE["classes"] = json.load(f)


def _prep_pil(img):
    arr = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return (arr - MEAN) / STD


def feats_from_pils(imgs, bs=32):
    out = []
    dino, device = STATE["dino"], STATE["device"]
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            batch = np.stack([_prep_pil(im) for im in imgs[i:i + bs]])
            batch = torch.from_numpy(batch).to(device)
            h = dino(pixel_values=batch).last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)  # CLS + mean of patch tokens
            out.append(f.cpu().numpy())
    return np.concatenate(out)


def predict_proba_pil(imgs):
    return STATE["clf"].predict_proba(feats_from_pils(imgs))


# ----------------------------------------------------------------------------
# Explanation: occlusion sensitivity (red = prediction depends on this region)
# ----------------------------------------------------------------------------
def occlusion_map(pil_img, proba_fn, patch=48, stride=24):
    img = pil_img.convert("RGB").resize((224, 224))
    base = proba_fn([img])[0]
    cls = int(base.argmax())

    positions = list(range(0, 224 - patch + 1, stride))
    if positions[-1] != 224 - patch:
        positions.append(224 - patch)

    gray = Image.new("RGB", (patch, patch), (124, 116, 104))
    occluded, coords = [], []
    for y in positions:
        for x in positions:
            im = img.copy()
            im.paste(gray, (x, y))
            occluded.append(im)
            coords.append((x, y))

    probs = proba_fn(occluded)[:, cls]
    drops = base[cls] - probs

    heat = np.zeros((224, 224))
    count = np.zeros((224, 224))
    for (x, y), d in zip(coords, drops):
        heat[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1
    heat = np.clip(heat / count, 0, None)
    heat = heat / (heat.max() + 1e-8)
    return img, heat, cls, base


def overlay_heatmap(img, heat, alpha=0.45):
    from matplotlib import cm
    colored = Image.fromarray((cm.jet(heat)[..., :3] * 255).astype(np.uint8))
    return Image.blend(img, colored, alpha)


# ----------------------------------------------------------------------------
# Gradio callback
# ----------------------------------------------------------------------------
def predict(img, cond, strength):
    if img is None:
        return None, None, None, "Please upload an image or take a webcam photo."

    mod = apply_condition(img, cond, strength)

    t0 = time.perf_counter()
    proba = predict_proba_pil([mod])[0]
    ms = (time.perf_counter() - t0) * 1000
    classes = STATE["classes"]
    cls = int(proba.argmax())

    _, heat, _, _ = occlusion_map(mod, predict_proba_pil)
    overlay = overlay_heatmap(mod, heat)

    notes = []
    if cond != "None":
        notes.append(f"SYNTHETIC condition applied: {cond} (strength {strength:.2f})")
    if np.array(mod.convert("L")).mean() < 30:
        notes.append("WARNING: the image is very dark, the gesture may be invisible.")
    if proba.max() < 0.6:
        notes.append("WARNING: low confidence, the prediction may be unreliable.")
    notes.append(f"Prediction: {classes[cls]} ({proba[cls] * 100:.1f}%)")
    notes.append(
        f"Inference time: {ms:.0f} ms | Device: {STATE['device_name']} | Model: DINOv2-small + LogReg"
    )
    label = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return label, mod, overlay, "\n".join(notes)


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Hand Gesture Recognition (DINOv2)") as demo:
        gr.Markdown(
            "# Hand Gesture Recognition: Peace / Stop / Thumbs Up\n"
            "DINOv2 + logistic regression. Upload an image or use the webcam. "
            "The stress-test generator creates **synthetic** modified images. "
            "Images are processed in memory and are not saved by this app."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Input image")
                cond = gr.Dropdown(
                    ["None", "Dark", "Backlight", "Motion blur", "Noise"],
                    value="None",
                    label="Stress-test generator (synthetic condition)",
                )
                strength = gr.Slider(0, 1, value=0.5, step=0.05, label="Condition strength")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                out_label = gr.Label(num_top_classes=3, label="Class probabilities")
                out_mod = gr.Image(label="Model input (SYNTHETIC if a condition is applied)")
                out_exp = gr.Image(label="Explanation: occlusion sensitivity (red = important)")
                out_info = gr.Textbox(label="Info", lines=5)
        btn.click(predict, [inp, cond, strength], [out_label, out_mod, out_exp, out_info])
    return demo


if __name__ == "__main__":
    load_models()
    # Set GRADIO_SHARE=1 to get a temporary public link (for example from Google Colab).
    build_demo().launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
