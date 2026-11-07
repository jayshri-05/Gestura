# GESTURA requirements audit

Reviewed against the supplied project brief on 2026-11-07. The existing visual style is being kept; this document records what works, what is partial, and what remains.

## Repository status

No existing GitHub repository or project checkout was available in this workspace, consistent with the user's clarification that they had trained the model before starting a repository. GESTURA was scaffolded here and the already-trained model integrated; existing source repository components could not be inspected or reused.

## Dataset and model

| Requirement | Status | Evidence / limit |
|---|---|---|
| Use an existing public dataset; never train on personal hand photos | Complete | Kaggle Sign Language MNIST was downloaded and verified. No personal photographs were used for training. |
| Verify source, license, labels, and file format | Complete | Kaggle source page identifies CC0, 28×28 grayscale CSV pixels, and 24 static classes A–I and K–Y. |
| Preprocess, split train/validation/test, train, and evaluate | Complete | Seeded stratified train/validation split from the supplied train CSV; separate supplied test CSV used once for final metrics. |
| Save model, label map, history, plots, confusion matrix, and classification report | Complete | Included under `backend/models` and `backend/evaluation`. |
| Report genuine results and supported language | Complete | Test accuracy 90.95%; macro F1 90.61%; weighted F1 91.03%. Supports static ASL fingerspelling only. J and Z are excluded. |
| Train dynamic signs or translate complete sentences | Not supported | The dataset and model classify single still images, not motion, words, grammar, or sentences. |

## Translator and camera

| Requirement | Status | Evidence / limit |
|---|---|---|
| Start/stop webcam after user permission; live feed and camera status | Partial | Browser camera capture, permission request, stop control, device selection, and live preview are present. The user’s screenshot confirms the stream displays. |
| Detect and draw hand landmarks continuously | Confirmed working in latest user screenshot | MediaPipe points and connecting lines are visible on the raised hand. |
| Pass a crop to the trained model and show confidence | Implemented, webcam classification is unreliable | The latest user screenshot shows the model labeling a hand incorrectly despite a displayed score around 91%. This confirms the static dataset model does not transfer reliably to this webcam crop. |
| Confidence threshold, temporal consensus, duplicate suppression | Changed to manual review | The screenshot showed a high-score wrong label, demonstrating that raw softmax confidence is not trustworthy on webcam images. Auto-append and confidence-based acceptance were removed. The user sees top three guesses and must explicitly add one; this reduces silent pollution of the text but does not improve the model's visual accuracy. |
| Translation controls (clear/new sentence, undo, copy, speak) | Partial | Controls are implemented for a string of recognized letters. “Undo” removes one letter because the model does not recognize words; spaces are user-inserted. Controls have not all been exercised end to end. |
| Recognition history and hand positioning instructions | Partial | Recent manually accepted guesses are shown. The camera panel advises one centered hand, palm toward camera, plain background, and steady lighting. |

## Learning module

| Requirement | Status | Evidence / limit |
|---|---|---|
| Alphabet/number/greetings/common-word categories and accurate references | Partial | ASL alphabet cards cover A–Z. Twenty-four still classes have local training samples; J/Z are shown in a locally packaged public-domain A–Z chart with motion arrows but are not model classes. Cards also link to Lifeprint / ASL University. Numbers, greetings, and vocabulary remain absent. |
| Flashcards, flip/reveal, learned state, navigation | Implemented for alphabet subset | A–Z prompts and sharp local SVG crops from the full public-domain alphabet chart; previous/next/reveal controls; progress saved in localStorage. |
| Practice with camera and model feedback | Partial / experimental | Select a supported letter and start the existing camera; UI compares the model guess with the target and clearly says feedback is not verified. |
| Quiz, score, and restart | Implemented for alphabet subset | Ten multiple-choice questions use sharp crops from the public-domain alphabet chart, with score, restart, and backend score validation. This quizzes letter image recognition, not signing ability. |

## Website, API, privacy, and accessibility

| Requirement | Status | Evidence / limit |
|---|---|---|
| Professional, restrained, responsive visual design | Implemented to current scope | Existing cream/charcoal/rust palette and responsive styling are preserved; added sections use the same style. |
| Home page sections and navigation | Mostly implemented | Navigation, hero with translate/learn calls to action, introduction, translator, learning, how-it-works, technology, Why GESTURA, model scope, and footer are present. |
| Backend endpoints | Partial | Health, model status, prediction, signs, learning categories, flashcards, and quiz score validation endpoints exist. Learning content remains intentionally limited to the alphabet. |
| Camera privacy and no recording | Implemented in current flow | Video stays in the browser; only 784 grayscale pixel values are sent to `127.0.0.1:8000`; no frames are saved. |
| Accessibility, keyboard operation, responsive, error handling | Partial / not fully audited | Labels, status text, error messages, and responsive CSS exist. A full keyboard, screen-reader, contrast, and device-size audit remains. |

## Verification performed

- The React production build completed successfully after adding local sign images, image quiz, both homepage calls to action, How it works, Technology and Why GESTURA sections, and manual prediction review.
- Python syntax compilation passed for `backend/main.py`.
- Local API checks passed: `/api/health`, `/api/model/status` (checkpoint loaded, test accuracy 0.9095), `/api/learning/categories`, `/api/learning/flashcards` (26 entries; correct still-image flags), and `/api/learning/quiz/submit` (80% response for the 8/10 sample request).
- Vite served the page plus local public-domain `asl-letter-A.svg`, `asl-letter-Z.svg`, and the full alphabet chart with HTTP 200. In-app flashcard and quiz screenshots were visually checked with sharp local vector artwork; the existing layout was preserved.
- The user's latest camera screenshot confirms the preview and hand landmarks work, but shows a wrong letter despite a high model score. Preview brightness and the transmitted crop now share a fixed 1.35 brightness and 1.05 contrast adjustment; this improves dim display only and does not establish webcam recognition accuracy. This is a camera-to-dataset domain shift and uncalibrated confidence problem; no dataset test metric proves webcam accuracy. Manual review is now required before appending a letter.
- Learning screens and new endpoints are implemented, and image source/license/model limits are called out in the UI and README. No personal images were used to train or tune the model.
- Full application tests (learning flows, all controls, browser/device combinations, responsive layouts, and real-world webcam accuracy) have not been completed.

## Next work, preserving the current interface

1. Improve webcam classification using suitable public camera-style images or a new landmark model, with signer-separated train/validation/test data and actual camera-domain evaluation. The installed CNN is not a reliable camera recognizer.
2. Add detailed handshape/movement directions to the A–Z cards and extend teaching to numbers and vocabulary only after source/license verification.
3. Finish the accessibility, device-size, browser permission, and webcam accuracy evaluation.
