# GESTURA

GESTURA is a new project workspace because no existing website repository was available. This is an initial app foundation, not the full product described in the original brief.

For the preconfigured single-service Docker deployment, follow [DEPLOY.md](DEPLOY.md). The production bundle serves the frontend and API from one HTTPS origin. Raw and processed training data are excluded from the downloadable project archive; the trained checkpoint and evaluation records are included.

## Current working scope

- A React/Vite project page that fetches actual model status and evaluation data from the API.
- A FastAPI service with `/api/health`, `/api/model/status`, `/api/signs`, and `/api/predict`.
- The trained PyTorch checkpoint, label map, signer-separated evaluation report, confusion matrix, and training history.
- The classifier supports **static ASL fingerspelling letters A–I and K–Y**. J and Z are excluded because they require motion. It does not recognize words, grammar, or sentences, and it does not support other sign languages.

## Model and dataset

The active camera model uses [ASL-HG: American Sign Language Hand Gesture Image Dataset](https://data.mendeley.com/datasets/j4y5w2c8w9/1), version 1, released under **CC BY 4.0**. The source contains 36,000 RGB JPEG images across A–Z and 0–9, collected from 10 volunteers with varied natural indoor and outdoor backgrounds. Each volunteer contributes 100 images per class. The dataset and license are described by the [official Mendeley record](https://data.mendeley.com/datasets/j4y5w2c8w9/1) and the accompanying [Data in Brief article](https://doi.org/10.1016/j.dib.2026.112492). Attribution: Pranto et al., “A comprehensive image dataset of American Sign Language hand gestures,” *Data in Brief* 65 (2026), 112492, DOI 10.1016/j.dib.2026.112492.

The model supports 24 static letters, A–I and K–Y. J and Z are excluded because they require motion; digits are not part of this classifier. The 876,640,391-byte archive was downloaded from the official Mendeley file endpoint. Its SHA-256 (`594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1`), ZIP CRC, class counts, signer counts, sample image decoding and MediaPipe hand detection were checked before training. Full verification details and the signer assignment are in `backend/evaluation/asl_hg_webcam/dataset_verification.json`.

The data split is by volunteer, not by randomly selected image: P7/P3/P2/P8/P5/P6/P9/P4 train; P1 validates and chooses the checkpoint; P10 is held out for testing. Seed 42 fixes the split. There are 19,183 retained training crops, 2,400 validation crops and 2,400 test crops. Seventeen training photos were skipped because MediaPipe could not find a hand; hand-detection coverage was 99.91% for train and 100% for validation and test. Preprocessing uses the same MediaPipe 21-landmark square crop and 1.5× hand-box padding as the browser, converts to RGB, resizes to 64×64 with Lanczos, and normalizes pixel values to [0,1]. Training augmentation applies random horizontal flips and moderate per-channel brightness/contrast changes. No personal hand photographs were used.

**Signer-held-out results:** **82.58% accuracy** and **79.49% macro F1** (1,982/2,400) on volunteer P10. Best validation accuracy was 68.88% on volunteer P1, showing sizeable signer variation. On P10, the weakest classes were F (0/100), T (0/100), R (4/100), and N (19/100); see the class report and confusion matrix before interpreting the overall score. This is one held-out volunteer from one dataset, not a universal webcam accuracy estimate. A preliminary full-frame run on P10 scored 55.92%; matching training crops to the camera crop improved that result. P10 was never used for training or checkpoint selection, but it was evaluated in that earlier baseline, so the final score is model-development evaluation rather than a pristine blind challenge. The run summary, class report, confusion matrix, crop-detection coverage and training history are in `backend/evaluation/asl_hg_webcam/`. Earlier Kaggle/Sign Language MNIST and mixed-dataset experiments are retained under `backend/evaluation/` as diagnostics; they are not the active camera model.

The `/api/predict` endpoint accepts exactly 12,288 integers: 64×64 row-major RGB pixels in the 0–255 range. MediaPipe Hand Landmarker runs in the browser, draws the existing landmarks, and sends only the matching square hand crop to the local FastAPI service. Webcam video is not recorded or stored. The interface layout is unchanged. A letter must agree across recent frames before it appears; the user still reviews and adds each letter manually.

Camera recognition remains experimental. The signer-held-out score does not measure every webcam, signer, lighting condition, camera distance or signing style. Softmax scores are not calibrated probabilities, and temporal voting can reduce flicker but cannot fix a consistent model error. The model only classifies isolated static fingerspelling; it does not translate words or sentences. No personal hand photographs were used for training or tuning.

## Learning features

- The Learn ASL area includes A–Z cards with sharp local vector crops from a public-domain chart; previous/next/reveal controls; locally saved learned-letter progress; a 10-question image quiz with score/restart; and experimental camera practice feedback. The model itself only supports A–I and K–Y; J/Z require movement.
- Educational scope is limited to ASL fingerspelling. The app does not yet contain the requested numbers, greetings, everyday vocabulary, or sign-language grammar lessons.
- Card and quiz images are per-letter SVG viewBox crops from one locally packaged [Wikimedia Commons ASL alphabet chart](https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet.svg). The file page marks it public domain; creator details are in `frontend/public/learning/asl/ATTRIBUTION.txt`. The full chart also appears in-app. Each per-letter SVG is a standalone crop of the source vector chart, so the browser renders the original vector paths sharply. The cards link to [ASL University / Lifeprint's fingerspelling reference](https://www.lifeprint.com/asl101/fingerspelling/abc.htm) for supplemental instruction.
- ASL is a full visual language with its own grammar, distinct from English; the classifier covers only alphabet handshapes. See the [National Association of the Deaf ASL overview](https://nad.org/knowledge-hub/american-sign-language/what-is-american-sign-language/).
- Camera practice reports whether the model's current guess matches the selected letter. This is model feedback, not a verified assessment of the user's signing; live webcam performance has not been validated across users.
- Additional endpoints: `GET /api/learning/categories`, `GET /api/learning/flashcards`, and `POST /api/learning/quiz/submit`.

## Run locally on Windows

Open a terminal in this `GESTURA` folder.

### Backend

```powershell
cd backend
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload --port 8001
```

The API is at `http://127.0.0.1:8001`; its interactive schema is at `http://127.0.0.1:8001/docs`.

To reproduce training after downloading the verified ASL-HG archive into `backend/data/raw/asl-hg-raw-v1.zip`, run from the project root:

```powershell
py -m pip install -r backend/requirements.txt
py backend/train_asl_hg_webcam.py
```

### Frontend

Open a second terminal in the `frontend` folder:

```powershell
npm install
npm run dev
```

Vite prints the local page URL (normally `http://localhost:5173`).

## Known limitations and remaining work

This is not a full sign-language translator. The camera classifier is still experimental and has not been validated across live webcam signers. The learning experience currently covers the fingerspelling alphabet only; it does not yet provide numbers, greetings, common words, or grammar. Its quiz is an image-recognition learning exercise; it does not validate ASL signing ability. A dynamic sequence model for J/Z and word signs is not implemented. Accessibility and cross-device end-to-end evaluation remain incomplete. Do not present any of these as completed capabilities.
