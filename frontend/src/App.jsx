import { useCallback, useEffect, useRef, useState } from 'react';
import { DrawingUtils, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const API = import.meta.env.DEV ? 'http://127.0.0.1:8001' : '';
const ASL_REFERENCE = 'https://www.lifeprint.com/asl101/fingerspelling/abc.htm';
const MODEL_LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const cropRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const busyRef = useRef(false);
  const lastInferenceRef = useRef(0);
  const predictionWindowRef = useRef([]);
  const [status, setStatus] = useState(null);
  const [apiState, setApiState] = useState('Checking model service…');
  const [cameraState, setCameraState] = useState('Camera off');
  const [handState, setHandState] = useState('No hand detected');
  const [cameraError, setCameraError] = useState('');
  const [cameraOn, setCameraOn] = useState(false);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState('');
  const [landmarkerReady, setLandmarkerReady] = useState(false);
  const [landmarkerState, setLandmarkerState] = useState('Hand tracker not started');
  const [prediction, setPrediction] = useState(null);
  const [sentence, setSentence] = useState('');
  const [recent, setRecent] = useState([]);
  const [learned, setLearned] = useState(() => JSON.parse(localStorage.getItem('gestura-learned-letters') || '[]'));
  const [cardIndex, setCardIndex] = useState(0);
  const [cardFlipped, setCardFlipped] = useState(false);
  const [quiz, setQuiz] = useState(null);
  const [quizQuestion, setQuizQuestion] = useState(0);
  const [quizScore, setQuizScore] = useState(0);
  const [quizChoice, setQuizChoice] = useState('');
  const [quizResult, setQuizResult] = useState(null);
  const [practiceTarget, setPracticeTarget] = useState('A');

  useEffect(() => {
    let active = true;
    fetch(`${API}/api/model/status`)
      .then((response) => { if (!response.ok) throw new Error(); return response.json(); })
      .then((data) => { if (active) { setStatus(data); setApiState(data.loaded ? 'Model ready' : 'Model unavailable'); } })
      .catch(() => { if (active) setApiState('Backend is not running'); });
    return () => { active = false; };
  }, []);

  const loadHandTracker = useCallback(async () => {
    if (landmarkerRef.current) return landmarkerRef.current;
    setLandmarkerState('Loading hand tracker…');
    const vision = await FilesetResolver.forVisionTasks('/wasm');
    const task = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/hand_landmarker.task' },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.4,
    });
    landmarkerRef.current = task;
    setLandmarkerReady(true);
    setLandmarkerState('Hand tracker ready');
    return task;
  }, []);

  const refreshDevices = useCallback(async () => {
    const list = (await navigator.mediaDevices.enumerateDevices()).filter((item) => item.kind === 'videoinput');
    setDevices(list);
    if (!deviceId && list[0]) setDeviceId(list[0].deviceId);
  }, [deviceId]);

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const ctx = overlayRef.current?.getContext('2d');
    if (ctx && overlayRef.current) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    setCameraOn(false);
    setCameraState('Camera off');
    setHandState('No hand detected');
    setPrediction(null);
  }, []);

  const runPrediction = useCallback(async (pixels) => {
    if (busyRef.current) return;
    busyRef.current = true;
    try {
      const response = await fetch(`${API}/api/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels }),
      });
      if (!response.ok) throw new Error(`Prediction service returned ${response.status}`);
      const result = await response.json();
      const window = [...predictionWindowRef.current, result].slice(-5);
      predictionWindowRef.current = window;
      const votes = new Map();
      for (const item of window) votes.set(item.sign, (votes.get(item.sign) || 0) + 1);
      const [stableLetter, stableVotes] = [...votes.entries()].sort((a, b) => b[1] - a[1])[0] || [];
      if (stableVotes >= 3) {
        const agreeing = window.filter((item) => item.sign === stableLetter);
        const confidence = agreeing.reduce((total, item) => total + item.confidence, 0) / agreeing.length;
        setPrediction({ ...agreeing.at(-1), confidence, temporally_stable: true });
      } else {
        setPrediction(null);
      }
    } catch (error) {
      setCameraError(error.message || 'Could not reach the prediction API.');
    } finally {
      busyRef.current = false;
    }
  }, []);

  const startLoop = useCallback(() => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const cropCanvas = cropRef.current;
    if (!video || !overlay || !cropCanvas || !landmarkerRef.current) return;
    const ctx = overlay.getContext('2d');
    const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
    const drawer = new DrawingUtils(ctx);
    let lastVideoTime = -1;

    const frame = () => {
      if (!streamRef.current || !video.videoWidth || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }
      if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
      }
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const now = performance.now();
        let result;
        try {
          result = landmarkerRef.current.detectForVideo(video, now);
        } catch (error) {
          console.error('MediaPipe hand tracking failed:', error);
          setHandState('Hand tracker error');
          setCameraError(`Hand tracking stopped: ${error.message || 'unknown MediaPipe error'}`);
          return;
        }
        const hand = result.landmarks?.[0];
        if (hand) {
          setHandState(`Hand detected · ${result.handednesses?.[0]?.[0]?.categoryName || 'hand'}`);
          drawer.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: '#82d9a3', lineWidth: 3 });
          drawer.drawLandmarks(hand, { color: '#ffffff', fillColor: '#df4f38', lineWidth: 1, radius: 4 });

          const xs = hand.map((point) => point.x * video.videoWidth);
          const ys = hand.map((point) => point.y * video.videoHeight);
          const boxW = Math.max(...xs) - Math.min(...xs);
          const boxH = Math.max(...ys) - Math.min(...ys);
          const side = Math.max(boxW, boxH) * 1.5;
          const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
          const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
          const cropSide = Math.max(1, Math.min(side, video.videoWidth, video.videoHeight));
          const sx = Math.max(0, Math.min(video.videoWidth - cropSide, centerX - cropSide / 2));
          const sy = Math.max(0, Math.min(video.videoHeight - cropSide, centerY - cropSide / 2));

          if (cropSide > 8 && now - lastInferenceRef.current >= 450 && !busyRef.current) {
            cropCtx.drawImage(video, sx, sy, cropSide, cropSide, 0, 0, 64, 64);
            const { data } = cropCtx.getImageData(0, 0, 64, 64);
            const pixels = new Array(64 * 64 * 3);
            for (let i = 0, j = 0; i < data.length; i += 4) {
              pixels[j++] = data[i];
              pixels[j++] = data[i + 1];
              pixels[j++] = data[i + 2];
            }
            lastInferenceRef.current = now;
            runPrediction(pixels);
          }
        } else {
          setHandState('No hand detected · center your hand');
          predictionWindowRef.current = [];
          setPrediction(null);
        }
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
  }, [runPrediction]);

  const startCamera = useCallback(async () => {
    setCameraError('');
    setHandState('Searching for a hand…');
    setCameraState('Requesting camera permission…');
    try {
      await loadHandTracker();
      if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview is unavailable.');
      const constraints = { audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) } };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      setCameraOn(true);
      setCameraState('Camera connected');
      await refreshDevices();
      startLoop();
    } catch (error) {
      stopCamera();
      const messages = {
        NotAllowedError: 'Camera access was denied. Allow camera access in your browser site settings, then try again.',
        NotFoundError: 'No camera was found. Connect a webcam and try again.',
        NotReadableError: 'The camera is busy in another app. Close that app and try again.',
      };
      setCameraState('Camera unavailable');
      setCameraError(messages[error.name] || error.message || 'Could not start the camera.');
    }
  }, [deviceId, loadHandTracker, refreshDevices, startLoop, stopCamera]);

  useEffect(() => {
    if (navigator.mediaDevices?.enumerateDevices) refreshDevices().catch(() => {});
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      landmarkerRef.current?.close();
    };
  }, [refreshDevices]);

  const clearSentence = () => setSentence('');
  const undo = () => setSentence((text) => text.slice(0, -1));
  const addPrediction = () => { if (!prediction) return; setSentence((text) => text + prediction.sign); setRecent((items) => [prediction.sign, ...items].slice(0, 8)); };
  const addSpace = () => setSentence((text) => text && !text.endsWith(' ') ? `${text} ` : text);
  const speak = () => { if (sentence.trim() && 'speechSynthesis' in window) window.speechSynthesis.speak(new SpeechSynthesisUtterance(sentence)); };
  const copy = async () => { if (sentence && navigator.clipboard) await navigator.clipboard.writeText(sentence); };

  const learningLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  const supportedLetters = status?.classes?.length ? status.classes : MODEL_LETTERS;
  const currentCard = learningLetters[cardIndex] || 'A';
  const currentModelSupportsCard = supportedLetters.includes(currentCard);
  const markLearned = () => {
    const next = learned.includes(currentCard) ? learned.filter((letter) => letter !== currentCard) : [...learned, currentCard];
    setLearned(next); localStorage.setItem('gestura-learned-letters', JSON.stringify(next));
  };
  const startQuiz = () => {
    const answers = [...supportedLetters].sort(() => Math.random() - 0.5).slice(0, 10);
    const pool = answers.map((answer) => ({ answer, image: `/learning/asl/asl-letter-${answer}.svg?v=20261107-4`, options: [...new Set([answer, ...supportedLetters.filter((letter) => letter !== answer).sort(() => Math.random() - 0.5).slice(0, 3)])].sort() }));
    setQuiz(pool); setQuizQuestion(0); setQuizScore(0); setQuizChoice('');
    setQuizResult(null);
  };
  const answerQuiz = (choice) => {
    if (quizChoice) return;
    setQuizChoice(choice);
    if (choice === quiz[quizQuestion].answer) setQuizScore((score) => score + 1);
  };
  const nextQuizQuestion = async () => {
    if (quizQuestion + 1 === quiz.length) {
      try {
        const response = await fetch(`${API}/api/learning/quiz/submit`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ total: quiz.length, correct: quizScore }) });
        if (response.ok) setQuizResult(await response.json());
      } catch { /* The local score remains available when the API is offline. */ }
      setQuizQuestion(quiz.length);
    } else {
      setQuizQuestion((index) => index + 1); setQuizChoice('');
    }
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#home" aria-label="Gestura home">GESTURA<span className="brand-dot">.</span></a>
        <nav aria-label="Main navigation"><a href="#translator">Sign to text</a><a href="#learn">Learn ASL</a><a href="#why">Why GESTURA</a><a href="#scope">About this model</a></nav>
        <span className={`service ${status?.loaded ? 'ready' : ''}`}><i />{apiState}</span>
      </header>

      <section className="hero" id="home">
        <div className="eyebrow"><span>01</span> A sign language learning project</div>
        <h1>Communication starts<br />with <em>understanding.</em></h1>
        <p className="lede">GESTURA is being built to make sign language easier to explore and practice. The first working component recognizes static ASL fingerspelling letters.</p>
        <div className="hero-actions"><a className="primary-link" href="#translator">Start translating <span aria-hidden="true">↘</span></a><a className="secondary-link" href="#learn">Start learning <span aria-hidden="true">↘</span></a></div>
        <div className="hero-note"><span className="rule" />ASL · Static letters · Early build</div>
      </section>

      <section className="model-section" id="translator">
        <div className="section-heading"><div><div className="eyebrow"><span>TRANSLATOR</span> Static fingerspelling</div><h2>Sign to text</h2></div><p>Hold one letter sign in view, review the model guess, then choose whether to add it to your text.</p></div>
        <div className="translator-grid">
          <div className="camera-panel">
            <div className="camera-head"><div><span className={`camera-indicator ${cameraOn ? 'on' : ''}`} />{cameraState}</div><span>{handState}</span></div>
            <div className="camera-stage">
              <video ref={videoRef} playsInline muted aria-label="Live camera preview" />
              <canvas ref={overlayRef} aria-hidden="true" />
              {!cameraOn && <div className="camera-empty"><div className="camera-outline">◎</div><p>Camera preview appears here</p><span>Choose Start camera to allow access</span></div>}
            </div>
            <canvas ref={cropRef} width="64" height="64" className="hidden-canvas" aria-hidden="true" />
            <div className="camera-controls">
              <div className="camera-select-wrap"><label htmlFor="camera-select">Camera</label><select id="camera-select" value={deviceId} onChange={(event) => setDeviceId(event.target.value)} disabled={cameraOn || devices.length < 2}><option value="">Default camera</option>{devices.map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>)}</select></div>
              {!cameraOn ? <button className="solid-button" onClick={startCamera} disabled={!status?.loaded}>{status?.loaded ? 'Start camera' : 'Waiting for model'}</button> : <button className="outline-button" onClick={stopCamera}>Stop camera</button>}
            </div>
            <div className="tracking-status"><span>{landmarkerReady ? 'Hand tracking ready' : landmarkerState}</span><span>{status?.loaded ? 'Classifier ready' : 'Classifier unavailable'}</span></div>
            {cameraError && <p className="camera-error" role="alert">{cameraError}</p>}
            <p className="privacy-note">For a clearer image, keep one hand fully in frame against a plain background, with the palm toward the camera and a light in front of your hand (not behind it). Frames are processed in your browser; only a 64×64 RGB hand crop is sent to the local API. Video is not recorded or stored.</p>
          </div>

          <aside className="translation-panel">
            <div className="translation-top"><span className="card-kicker">EXPERIMENTAL CAMERA GUESS</span><div className="prediction-letter">{prediction?.sign || '—'}</div><div className="confidence-line"><span>Stable model score (not calibrated)</span><strong>{prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : '—'}</strong></div><div className="confidence-meter"><span style={{ width: `${Math.max(0, Math.min(100, (prediction?.confidence || 0) * 100))}%` }} /></div><p className="threshold-note">A letter is shown after it agrees across recent frames. Review every guess before adding it.</p><div className="top-alternatives">{prediction?.top_predictions?.slice(0, 3).map((item) => <span key={item.sign}>{item.sign} · {(item.confidence * 100).toFixed(0)}%</span>)}</div><button className="add-guess" onClick={addPrediction} disabled={!prediction}>Add this letter</button></div>
            <div className="recognized-block"><div className="card-kicker">YOUR LETTERS</div><div className="sentence-output" aria-live="polite">{sentence || <span className="placeholder">Add reviewed letters here</span>}</div><div className="sentence-actions"><button onClick={addSpace} disabled={!sentence}>Space</button><button onClick={undo} disabled={!sentence}>Undo</button><button onClick={copy} disabled={!sentence}>Copy</button><button onClick={speak} disabled={!sentence}>Speak</button><button onClick={clearSentence} disabled={!sentence}>Clear text</button></div></div>
            <div className="recent-block"><div className="card-kicker">RECENTLY ADDED LETTERS</div><div className="recent-letters">{recent.length ? recent.map((letter, index) => <span key={`${letter}-${index}`}>{letter}</span>) : <span className="placeholder">No letters added yet</span>}</div></div>
          </aside>
        </div>
        <div className="api-note" role="note"><span className="note-mark">i</span><p>This model recognizes isolated static ASL fingerspelling letters only (A–I, K–Y). It uses RGB camera crops and was checked against a separate public image set; that score is not a guarantee of webcam accuracy. Lighting, hand orientation, framing and signer differences still affect results. Review guesses before adding them. J and Z require movement and are not recognized by this still-image model.</p></div>
      </section>

      <section className="learn-section" id="learn">
        <div className="section-heading"><div><div className="eyebrow"><span>03</span> Learn sign language</div><h2>Practice ASL fingerspelling</h2></div><p>Study A–Z from the linked ASL reference. J and Z require movement and are not supported by this still-image classifier.</p></div>
        <div className="learning-grid">
          <article className="study-panel">
            <div className="card-kicker">LETTER FLASHCARDS · {cardIndex + 1} OF {learningLetters.length || 24}</div>
            <div className={`flashcard ${cardFlipped ? 'flipped' : ''}`}>
              <img className="sign-sample" src={`/learning/asl/asl-letter-${currentCard}.svg?v=20261107-4`} alt={`High-resolution ASL fingerspelling ${currentCard} handshape from the public-domain alphabet chart`} />
              <span className="flashcard-letter">{currentCard}</span>
              {!currentModelSupportsCard ? <><span className="flashcard-prompt">J and Z require movement and are not recognized by the still-image model.</span><a href={ASL_REFERENCE} target="_blank" rel="noreferrer">Open ASL University motion-letter reference ↗</a></> : cardFlipped ? <><p>This handshape comes from the labeled public-domain alphabet chart. Signer variations exist; use the full chart below for context.</p><a href={ASL_REFERENCE} target="_blank" rel="noreferrer">Open ASL University alphabet reference ↗</a></> : <span className="flashcard-prompt">Public-domain handshape crop · Select “Reveal reference” for notes.</span>}
            </div>
            <div className="study-actions"><button disabled={!learningLetters.length} onClick={() => setCardIndex((index) => (index - 1 + learningLetters.length) % learningLetters.length)}>Previous</button><button disabled={!learningLetters.length} onClick={() => setCardFlipped((flipped) => !flipped)}>{cardFlipped ? 'Hide reference' : 'Reveal reference'}</button><button disabled={!learningLetters.length} onClick={markLearned}>{learned.includes(currentCard) ? 'Learned ✓' : 'Mark learned'}</button><button disabled={!learningLetters.length} onClick={() => { setCardIndex((index) => (index + 1) % learningLetters.length); setCardFlipped(false); }}>Next</button></div>
            <p className="learning-progress">Progress saved on this device: {learned.length} of {learningLetters.length} alphabet letters marked learned. Public dataset images are available for {supportedLetters.length} still letters.</p>
          </article>
          <article className="quiz-panel"><div className="card-kicker">IMAGE QUIZ · STILL LETTERS</div>{!quiz ? <><h3>Quick quiz</h3><p>Identify the letter from a sharp handshape crop in the public-domain alphabet chart. J and Z are excluded because they require movement.</p><button className="solid-button" onClick={startQuiz} disabled={!supportedLetters.length}>Start 10-question quiz</button></> : quizQuestion >= quiz.length ? <><h3>Quiz complete</h3><p>Your score: {quizScore} of {quiz.length}{quizResult ? ` · ${quizResult.score_percent}%` : ''}</p><button className="solid-button" onClick={startQuiz}>Restart quiz</button></> : <><p className="quiz-count">Question {quizQuestion + 1} of {quiz.length} · Which letter is shown?</p><img className="quiz-sign-sample" src={quiz[quizQuestion].image} alt="Handshape crop from the public-domain ASL alphabet chart" /><div className="quiz-options">{quiz[quizQuestion].options.map((letter) => <button key={letter} className={quizChoice === letter ? (letter === quiz[quizQuestion].answer ? 'correct' : 'incorrect') : ''} onClick={() => answerQuiz(letter)} disabled={Boolean(quizChoice)}>{letter}</button>)}</div>{quizChoice && <><p className="quiz-feedback">{quizChoice === quiz[quizQuestion].answer ? 'Correct.' : `Not quite. The answer is ${quiz[quizQuestion].answer}.`} This is one visual chart example; signing can vary.</p><button onClick={nextQuizQuestion}>Next question</button></>}</>}</article>
          <article className="practice-panel"><div className="card-kicker">CAMERA PRACTICE · EXPERIMENTAL</div><h3>Try a letter</h3><label htmlFor="practice-target">Practice target</label><select id="practice-target" value={practiceTarget} onChange={(event) => setPracticeTarget(event.target.value)}>{supportedLetters.map((letter) => <option key={letter}>{letter}</option>)}</select><button className="solid-button" onClick={cameraOn ? stopCamera : startCamera} disabled={!cameraOn && !status?.loaded}>{cameraOn ? 'Stop practice camera' : 'Start practice camera'}</button><p>Camera preview is in the translator panel above. The model currently predicts <strong>{prediction?.sign || '—'}</strong>.</p>{prediction && <p className={prediction.sign === practiceTarget ? 'practice-match' : 'practice-mismatch'}>{prediction.sign === practiceTarget ? 'The model guess matches your target.' : `The model guess does not match ${practiceTarget}.`}</p>}<p className="practice-caveat">This is model feedback only, not a verified assessment of your handshape. Camera recognition is not yet validated.</p><a href={ASL_REFERENCE} target="_blank" rel="noreferrer">Open handshape reference ↗</a></article>
        </div>
        <div className="alphabet-chart"><div><div className="card-kicker">VISUAL REFERENCE · A–Z</div><h3>ASL manual alphabet chart</h3><p>Full handshape chart, including the motion paths for J and Z. Dataset samples above remain separate from this educational chart.</p><a href="https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet.svg" target="_blank" rel="noreferrer">Image source and public-domain license ↗</a></div><img src="/learning/asl/asl-alphabet-gallaudet.svg" alt="Public-domain American Sign Language fingerspelling alphabet chart showing labeled handshapes A through Z" loading="lazy" /></div>
        <p className="learning-source">Learning scope: ASL manual alphabet practice only, not ASL vocabulary or grammar. Letter cards and quiz use high-resolution crops from the locally included public-domain chart; the webcam model uses public labeled photo datasets documented in the project README. The chart includes J/Z motion paths; those classes are absent from the still-image model. Reference: <a href={ASL_REFERENCE} target="_blank" rel="noreferrer">ASL University / Lifeprint handshapes</a>.</p>
      </section>

      <section className="how-section" id="how-it-works"><div className="eyebrow"><span>04</span> How it works</div><div className="how-grid"><article><strong>01</strong><h3>Track the hand</h3><p>MediaPipe finds a hand and draws its 21 landmarks in your browser.</p></article><article><strong>02</strong><h3>Classify a still</h3><p>A local PyTorch model predicts among 24 static ASL fingerspelling classes. Webcam guesses are experimental.</p></article><article><strong>03</strong><h3>Review the result</h3><p>You choose whether to add a guessed letter. No word-level ASL translation is claimed.</p></article></div><p className="how-foot">Built with React, MediaPipe Tasks Vision, FastAPI and PyTorch. Camera video stays in the browser; only a 64×64 RGB hand crop is sent to the local API.</p></section>

      <section className="why-section" id="why"><div className="eyebrow"><span>05</span> Why GESTURA</div><div className="why-content"><h2>Trust starts with<br />clear limits.</h2><div><p>Sign languages are full languages with their own grammar and communities. A fingerspelling classifier is only one small learning tool; it cannot replace learning ASL or interpreting a signed conversation. <a href="https://nad.org/knowledge-hub/american-sign-language/what-is-american-sign-language/" target="_blank" rel="noreferrer">Learn about ASL from the National Association of the Deaf ↗</a></p><p>GESTURA keeps camera video in the browser, names the source of its training examples, and separates benchmark scores from unverified webcam behavior. The aim is to make the demo useful without overstating what it understands.</p></div></div></section>

      <section className="scope" id="scope">
        <div className="eyebrow"><span>02</span> About this model</div>
        <div className="scope-content"><h2>Measured on<br />held-out data.</h2><div><p className="result-stat">{status?.test_accuracy != null ? `${(status.test_accuracy * 100).toFixed(2)}%` : '—'} <small>test accuracy</small></p><p>Measured on the separate 7,172-image Kaggle test split. Macro F1: {status?.test_macro_f1 != null ? `${(status.test_macro_f1 * 100).toFixed(2)}%` : '—'}.</p><p>The random validation split reached 100%; augmented examples may be correlated across that split, so the held-out test result is the more useful benchmark.</p></div></div>
      </section>

      <footer><span className="brand">GESTURA<span className="brand-dot">.</span></span><span>Camera remains local; no video is stored.</span><span>ASL static letters · v0.4</span></footer>
    </main>
  );
}
