# Quick guide: Deploy to Hugging Face Spaces

This short guide shows how to deploy the project to Hugging Face Spaces using the contents in `huggingface_space/`.

Two simple options are covered:
- Option A: Gradio Web App (point-and-click UI)
- Option B: FastAPI REST API (programmatic use)

> Tip: Create a dedicated Space with only the files from `huggingface_space/` at the repo root. That matches Spaces’ default expectations for the app file and requirements.

---

## 0) Prepare the Space folder

From this repo, the production-ready assets are here:

```
huggingface_space/
├─ app.py                 # Gradio web UI
├─ api_app.py             # FastAPI app (exposes `app`)
├─ requirements.txt       # Dependencies for the Space
├─ start.py               # Local helper (not required by Spaces)
├─ xgboost_retrained_tuned.pkl
└─ preprocessing/
   ├─ label_encoders.pkl
   └─ scaler.pkl
```

Make sure the three model files exist in `huggingface_space/`:
- `xgboost_retrained_tuned.pkl`
- `preprocessing/label_encoders.pkl`
- `preprocessing/scaler.pkl`

If your latest models live under `/models`, copy the correct versions into `huggingface_space/` before deploying.

---

## Option A — Gradio Space (recommended for demos)

1) Create a new Space on Hugging Face
- Go to https://huggingface.co/spaces → “Create new Space”
- Space SDK: “Gradio”
- Visibility: Public or Private

2) Push the app files
- Place the contents of `huggingface_space/` at the root of your new Space repository (so `app.py` and `requirements.txt` are in the top level).
- Commit and push via Git, or drag-and-drop files in the Spaces UI.

3) Build & run
- Spaces will auto-install from `requirements.txt` and run `app.py`.
- When the build completes, the UI will be live.

4) Use it
- Open the Space URL; you’ll see the Gradio form.
- Make a prediction by filling inputs and clicking submit.

Common notes
- No ports to configure—Gradio on Spaces handles that.
- If you change model files, re-upload and push to trigger a rebuild.

---

## Option B — FastAPI Space (for REST APIs)

1) Create a new Space
- Space SDK: “FastAPI”

2) Push the API files
- Put the contents of `huggingface_space/` at the root of the Space.
- Ensure `api_app.py` exists at the repo root (or set the Space’s `app_file` to `api_app.py`).

3) Build & run
- Spaces will serve the FastAPI app automatically.
- API docs will be available at `/docs` on the Space URL.

4) Test the API (example)

Single prediction (adjust your Space URL):
```python
import requests

payload = {
  "age": 35,
  "job": "management",
  "marital": "married",
  "education": "university.degree",
  "balance": 1500
  # ... other features
}

resp = requests.post("https://<your-namespace-your-space>.hf.space/predict", json=payload)
print(resp.json())
```

Common notes
- With Spaces’ FastAPI template, you typically don’t need to run Uvicorn manually. Spaces will serve the `app` object from `api_app.py`.
- If you do manage your own server, ensure it binds to the `PORT` env var: `int(os.environ.get("PORT", 7860))`.

---

## Troubleshooting

- Missing model files
  - If the app warns about missing `xgboost_retrained_tuned.pkl` or preprocessors, copy them into `huggingface_space/` and re-deploy.

- Large files
  - If model files exceed Git size limits, use Git LFS on your Space repo.

- Dependency errors
  - Pin versions in `huggingface_space/requirements.txt`. Rebuild by pushing a commit.

- App file path
  - The app entry must be at the Space root (`app.py` for Gradio or `api_app.py` for FastAPI). If you keep them in a subfolder, configure the Space’s `app_file` accordingly.

---

## Local run (optional)

You can run locally before pushing:

- Gradio:
```bash
cd huggingface_space
python app.py
```

- FastAPI (Uvicorn):
```bash
cd huggingface_space
uvicorn api_app:app --host 0.0.0.0 --port 7860
```

Or use the helper script:
```bash
cd huggingface_space
python start.py --mode gradio   # or --mode fastapi
```

---

That’s it—you now have a simple, repeatable path to deploy the project as a Gradio app or a FastAPI API on Hugging Face Spaces.
