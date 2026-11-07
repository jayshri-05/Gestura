**GESTURA**
GESTURA is a web app designed to help users learn ASL fingerspelling and explore camera-based sign recognition. It combines interactive learning tools with an experimental AI-powered camera feature.

Live Demo: Open GESTURA
**Key Features**
- Learn ASL: Interactive A–Z flashcards, quizzes, and progress tracking.
- Sign Recognition: Uses a trained PyTorch model to recognize 24 static ASL letters (A–I, K–Y) through a webcam.
- Live Hand Tracking: Uses MediaPipe to detect hand landmarks and capture hand movements.
- Backend API: Built with FastAPI to handle predictions and model information.
- Privacy: Webcam footage is not recorded or stored.
**Model & Performance**
- Trained using the publicly available ASL-HG dataset.
- Achieved 82.58% accuracy and 79.49% macro F1 on a separate signer test set.
- J and Z are excluded because they require movement.
- Recognition is experimental and may vary across users and lighting conditions.
**Technology Stack**
React, Vite, Python, FastAPI, PyTorch, and MediaPipe.
**Current Limitations**
GESTURA currently focuses on ASL alphabet learning and static letter recognition. It does not yet support numbers, everyday words, greetings, sentences, or ASL grammar. Full sign-language translation is still future work.
**Run Locally**
- Backend: Install dependencies and run FastAPI on port 8001.
- Frontend: Install npm packages and start the Vite development server.
See DEPLOY.md for deployment instructions.

GESTURA aims to make learning ASL more interactive and accessible while exploring the possibilities of real-time sign recognition.
