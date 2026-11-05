# Hand Gesture Recognition: Peace / Stop / Thumbs Up (Assignment 2)

Computer Vision, Astana IT University. Student: Berdimbet Alima (ST2405). Instructor: Nurym Kaiyrkhan.

Continuation of Assignment 1 (self-collected dataset of 372 hand-gesture photos, baseline CNN and MobileNetV2).
In Assignment 2 the same topic is upgraded with a SOTA component and an interactive demo:

- **SOTA component:** DINOv2-small (self-supervised Vision Transformer, frozen) + logistic regression.
- **Stress test:** 96 new real photos (dark, backlight, motion blur, new background), used for evaluation only.
- **Synthetic conditions:** programmatic dark / backlight / blur / noise / crop variants of the training photos (1,040 images, 80% of the training set). All synthetic images are labelled SYNTHETIC.
- **Explainability:** occlusion sensitivity (main) and attention rollout (shown as an unreliable alternative).
- **Demo app:** Gradio, upload or webcam, stress-test generator, probabilities, explanation, latency and device.

## Results

Original test set (56 images, same split and seed as A1):

| Model | Accuracy (%) | Macro F1 (%) |
|---|---|---|
| A1 baseline CNN | 83.93 | 83.93 |
| A1 MobileNetV2 (fine-tuned) | 80.36 | 79.56 |
| DINOv2 + LogReg | 100.00 | 100.00 |

This test is small and similar to the training photos, so the 100% is probably optimistic.

Stress test (96 photos, accuracy in %, chance = 33%):

| Model | Backlight | Blur | Dark | New background | Overall |
|---|---|---|---|---|---|
| A1 baseline CNN | 39.1 | 28.0 | 27.3 | 46.2 | 35.4 |
| DINOv2 + LogReg (real only) | 39.1 | 36.0 | 40.9 | 53.8 | 42.7 |
| DINOv2 + LogReg (+ synthetic) | 39.1 | 36.0 | 54.5 | 65.4 | 49.0 |

With 96 images the 95% confidence interval is roughly +/-10 percentage points, so small differences are suggestive, not conclusive.
Latency on a Colab T4: about 24 ms per prediction in the app (about 50 ms including file reading), about 689 ms for the occlusion explanation.

## Repository structure

```
app.py                          Gradio demo
requirements.txt
dinov2_logreg_synth.joblib      final classifier (trained with synthetic conditions)
dinov2_logreg_real_only.joblib  ablation classifier (real images only)
class_names.json                ["peace", "stop", "thumbs_up"]
notebooks/                      A1 notebook and A2 notebook (Colab)
results/                        figures and csv files (stress test, ablation, learning curve, explanations)
report/                         Assignment2_Report_Berdimbet_Alima.pdf
```

The raw dataset (photos of my own hands, some backgrounds partly show other people) is **not** included in this repository.
To reproduce the experiments with your own photos use the folder layout below.

## How to run

### Option 1: demo app (local or Colab)

```bash
pip install -r requirements.txt
python app.py
```
Open the printed local address. On Google Colab use `GRADIO_SHARE=1 python app.py` (or `demo.launch(share=True)`) to get a temporary public link.
The DINOv2 weights (`facebook/dinov2-small`) are downloaded automatically from Hugging Face. Keep `dinov2_logreg_synth.joblib` and `class_names.json` next to `app.py`.
A GPU is optional: on CPU the prediction works but the explanation is slower.

### Option 2: notebook (Google Colab, free T4 GPU)

1. Put your data in Google Drive: `Computer_Vision_Assignment/hand_gesture_dataset/<class>/*.jpg` and
   `Computer_Vision_Assignment/stress_test/<condition>/<class>/*.jpg` (conditions: `dark`, `backlight`, `blur`, `new_bg`; classes: `peace`, `stop`, `thumbs_up`).
2. Open the A2 notebook in Colab, set Runtime to T4 GPU and run the cells in order.
3. The notebook saves the classifiers, split files (`split_train.csv`, `split_val.csv`, `split_test.csv`), figures and csv results back to Drive. Random seed: 42.

## Ethics and limitations (short)

- Synthetic images are simple edits of my own photos, used for training only and always labelled SYNTHETIC. No generated faces or avatars.
- The data comes mostly from one person's hands and a few places; performance for other people, cameras and backgrounds is unknown and expected to be worse.
- Predictions can be wrong with very high confidence (see the report). Do not use this model for decisions about people or in safety-critical settings.
- The app processes images in memory and does not intentionally save them.
- The Step-Video-T2V and ArtAug generators from the assignment table were not used because they do not fit free Colab; DINOv2 was chosen as another modern SOTA.

## References

1. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193, 2023.
2. Darcet et al. Vision Transformers Need Registers. arXiv:2309.16588, 2023.
3. Zeiler, Fergus. Visualizing and Understanding Convolutional Networks. ECCV, 2014.
4. Abnar, Zuidema. Quantifying Attention Flow in Transformers. ACL, 2020.
