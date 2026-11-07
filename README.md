# GESTURA

**Learn ASL. Explore Sign Recognition.**

GESTURA is a web app that helps users learn American Sign Language (ASL) fingerspelling and explore camera-based sign recognition. It combines interactive learning tools with an experimental AI-powered camera feature.

🌐 **Live Demo:** [Open GESTURA](https://gestura-5oxt.onrender.com/)

---

## ✨ Key Features

- **Learn ASL:** Interactive A–Z flashcards, quizzes, and learning progress tracking.
- **Sign Recognition:** Uses a trained PyTorch model to recognize 24 static ASL letters (A–I and K–Y) through a webcam.
- **Live Hand Tracking:** Uses MediaPipe to detect hand landmarks and track hand positions.
- **Backend API:** Built with FastAPI to handle predictions and model information.
- **Privacy:** Webcam footage is not recorded or stored.

---

## 🤖 Model & Performance

- Trained using the publicly available ASL-HG dataset.
- Achieved **82.58% accuracy** and **79.49% macro F1-score** on a separate signer test set.
- Supports 24 static letters, excluding J and Z because they require movement.
- Recognition is experimental and may vary depending on users, lighting, and camera conditions.

---

## 💻 Technology Stack

- **Frontend:** React, Vite
- **Backend:** Python, FastAPI
- **Machine Learning:** PyTorch
- **Hand Tracking:** MediaPipe

---

## ⚠️ Current Limitations

GESTURA currently focuses on ASL alphabet learning and static letter recognition. It does not yet support numbers, greetings, everyday words, sentences, or ASL grammar.

The camera recognition feature is still experimental and is not a complete sign-language translation system.

---

## 🚀 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/jayshri-05/Gestura.git
cd Gestura
```

**2. Start the Backend**

```bash
cd backend
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload --port 8001
```

**3. Start the Frontend**

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL provided by Vite (usually `http://localhost:5173`).

---

## 📌 Future Improvements

- Dynamic sign recognition for J and Z.
- Support for numbers and everyday signs.
- Word and sentence-level recognition.
- Improved accuracy across different users and environments.
- Better accessibility and cross-device support.

---

**GESTURA aims to make ASL learning more interactive and accessible while exploring the possibilities of real-time sign recognition.**
