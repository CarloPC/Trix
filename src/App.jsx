import { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import {
  loadFaceModels,
  getAllFaceDescriptors,
  fetchKnownFaces,
  matchFace,
} from './lib/faceRecognition';
import './index.css';

const FACE_SCORE_THRESHOLD = 0.5;
const MOTION_THRESHOLD = 15;
const MISS_TOLERANCE = 5;
const ROBOT_NAME = 'Trix';
const INTRO_GREETING = `Hi there! I'm ${ROBOT_NAME}, your friendly robot assistant. Nice to meet you!`;
const DEFAULT_GREETING = 'Welcome, visitor!';
// How long, per person (or per "unknown visitor"), before we're willing to
// greet them again if they're still lingering in frame.
const RE_GREET_COOLDOWN_MS = 15000;
// How long to stay "awake" after the last face/speech activity before
// drifting off to sleep.
const SLEEP_GRACE_MS = 5000;
// How often (ms) to scan for faces once someone is present. Kept separate
// from the 500ms presence/motion loop since it's a heavier call. Short
// enough that the confirm-count check below still resolves in well under
// a second.
const RECOGNITION_INTERVAL_MS = 350;
// Identity confirmation now uses a hit/miss SCORE per candidate rather than
// a strict "must match every single consecutive scan" streak. A scan that
// doesn't see a given identity subtracts a small penalty instead of wiping
// their progress back to zero -- so one noisy/ambiguous frame (a blink, a
// bad angle, momentary occlusion by another person in frame) doesn't force
// them to start over. A candidate is only dropped once it's been missed for
// several scans in a row, i.e. they've genuinely left frame.
//
// Confirming a KNOWN face and confirming "unknown visitor" are NOT held to
// the same bar on purpose. Greeting someone a scan late is a minor delay;
// confidently telling a known person "Welcome, visitor!" is the costly,
// embarrassing mistake. So it takes far more accumulated evidence to
// confirm "visitor" than to confirm a recognized face.
const RECOGNITION_CONFIRM_HITS_KNOWN = 3;
const RECOGNITION_CONFIRM_HITS_UNKNOWN = 8;
// How much a candidate's score decays on a scan where it isn't seen.
const RECOGNITION_MISS_PENALTY = 1;
// A candidate not seen for this many consecutive scans is dropped entirely
// (genuinely gone, not just a glitchy frame).
const RECOGNITION_DROP_AFTER_MISSES = 6;
// After finishing a greeting, how long to keep showing that person's name
// on screen before handing off to the next person in the queue.
const GREET_DISPLAY_HOLD_MS = 1200;
const UNKNOWN_KEY = 'UNKNOWN_VISITOR';

// Eye-tracking tuning
const MAX_PUPIL_OFFSET_X = 22;
const MAX_PUPIL_OFFSET_Y = 14;
const INVERT_X = true;
const INVERT_Y = false;

// Blink tuning
const BLINK_MIN_DELAY = 2200;
const BLINK_MAX_DELAY = 5000;
const BLINK_DURATION = 160;

function RobotFace({ state, isSpeaking, eyeOffset, isBlinking }) {
  const { x, y } = eyeOffset;

  return (
    <svg className="face-svg" viewBox="0 0 240 140" xmlns="http://www.w3.org/2000/svg">
      {state === 'loading' && (
        <g className="face-group loading">
          <circle className="loading-dot" cx="90" cy="70" r="7" />
          <circle className="loading-dot delay1" cx="120" cy="70" r="7" />
          <circle className="loading-dot delay2" cx="150" cy="70" r="7" />
        </g>
      )}

      {state === 'sleeping' && (
        <g className="face-group sleeping">
          <path className="eye eye-left" d="M 60 70 Q 80 82 100 70" />
          <path className="eye eye-right" d="M 140 70 Q 160 82 180 70" />
          <path className="mouth" d="M 105 95 Q 120 90 135 95" />
          <text className="z-particle z1" x="165" y="40">Z</text>
          <text className="z-particle z2" x="180" y="25">Z</text>
          <text className="z-particle z3" x="195" y="10">z</text>
        </g>
      )}

      {state === 'happy' && (
        <g className="face-group happy">
          <g
            className={`eye-unit eye-unit-left ${isBlinking ? 'blinking' : ''}`}
            style={{ transform: `translate(${x}px, ${y}px)` }}
          >
            <circle className="eye-white" cx="80" cy="65" r="22" />
            <circle className="pupil" cx="80" cy="65" r="10" />
            <circle className="pupil-shine" cx="77" cy="62" r="3" />
          </g>

          <g
            className={`eye-unit eye-unit-right ${isBlinking ? 'blinking' : ''}`}
            style={{ transform: `translate(${x}px, ${y}px)` }}
          >
            <circle className="eye-white" cx="160" cy="65" r="22" />
            <circle className="pupil" cx="160" cy="65" r="10" />
            <circle className="pupil-shine" cx="157" cy="62" r="3" />
          </g>

          <g
            className="mouth-unit"
            style={{ transform: `translate(${x * 0.4}px, ${y * 0.3}px)` }}
          >
            <path
              className={`mouth smile ${isSpeaking ? 'talking' : ''}`}
              d="M 95 107 Q 120 133 145 107"
            />
          </g>
        </g>
      )}
    </svg>
  );
}

function App() {
  const webcamRef = useRef(null);
  const modelRef = useRef(null);
  const intervalRef = useRef(null);
  const recognitionIntervalRef = useRef(null);
  const blinkTimeoutRef = useRef(null);
  const sleepTimeoutRef = useRef(null);
  const canvasRef = useRef(document.createElement('canvas'));
  const prevFrameRef = useRef(null);
  const missCountRef = useRef(0);
  const isRecognizingRef = useRef(false);

  // --- Multi-face greeting queue -------------------------------------
  // When several people are in frame at once, we don't try to greet them
  // all simultaneously (that's what was getting "scrambled"). Instead each
  // confirmed identity is pushed onto a queue and greeted one at a time.
  const greetQueueRef = useRef([]); // pending entries: { key, name, title }
  const queuedKeysRef = useRef(new Set()); // keys currently queued OR being greeted
  const activeGreetRef = useRef(null); // entry currently being greeted, if any
  const candidateStreakRef = useRef(new Map()); // key -> { score, misses } (see RECOGNITION_* constants)
  const lastGreetedAtRef = useRef(new Map()); // key -> timestamp last greeted

  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voicesReady, setVoicesReady] = useState(false);
  const [eyeOffset, setEyeOffset] = useState({ x: 0, y: 0 });
  const [isBlinking, setIsBlinking] = useState(false);
  // The person (or "unknown visitor") currently being greeted / displayed.
  // null means nobody is actively being addressed right now.
  const [activeGreeting, setActiveGreeting] = useState(null);
  const [knownFaces, setKnownFaces] = useState([]);
  const [hasStarted, setHasStarted] = useState(false);
  const [introFinished, setIntroFinished] = useState(false);
  const [isAwake, setIsAwake] = useState(false);

  // Load BlazeFace (presence/wake detection + eye tracking)
  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const loadedModel = await blazeface.load({
        maxFaces: 1,
        scoreThreshold: FACE_SCORE_THRESHOLD,
      });
      modelRef.current = loadedModel;
      setIsModelLoading(false);
    };
    loadModel();
  }, []);

  // Load face recognition models + the known-faces cache from Supabase
  useEffect(() => {
    loadFaceModels();
    fetchKnownFaces().then((faces) => {
      console.log(`Loaded ${faces.length} known face(s) from Supabase.`);
      setKnownFaces(faces);
    });
  }, []);

  // Voices for speech synthesis
  useEffect(() => {
    if (!('speechSynthesis' in window)) return;
    const handleVoices = () => {
      if (window.speechSynthesis.getVoices().length > 0) setVoicesReady(true);
    };
    handleVoices();
    window.speechSynthesis.onvoiceschanged = handleVoices;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  // Stay "awake" as long as a face is present or she's speaking. Once both
  // go false, wait SLEEP_GRACE_MS of continued inactivity before sleeping --
  // any face/speech activity during that window cancels the countdown.
  useEffect(() => {
    const activelyAwake = isFaceDetected || isSpeaking;

    if (activelyAwake) {
      if (sleepTimeoutRef.current) {
        clearTimeout(sleepTimeoutRef.current);
        sleepTimeoutRef.current = null;
      }
      setIsAwake(true);
    } else if (!sleepTimeoutRef.current) {
      sleepTimeoutRef.current = setTimeout(() => {
        setIsAwake(false);
        sleepTimeoutRef.current = null;
      }, SLEEP_GRACE_MS);
    }
  }, [isFaceDetected, isSpeaking]);

  // Clear any pending sleep timer on unmount
  useEffect(() => {
    return () => {
      if (sleepTimeoutRef.current) clearTimeout(sleepTimeoutRef.current);
    };
  }, []);

  // Independent blink loop -- runs whenever she's awake
  useEffect(() => {
    if (!isAwake) return;

    const scheduleBlink = () => {
      const delay = BLINK_MIN_DELAY + Math.random() * (BLINK_MAX_DELAY - BLINK_MIN_DELAY);
      blinkTimeoutRef.current = setTimeout(() => {
        setIsBlinking(true);
        setTimeout(() => setIsBlinking(false), BLINK_DURATION);
        scheduleBlink();
      }, delay);
    };

    scheduleBlink();
    return () => clearTimeout(blinkTimeoutRef.current);
  }, [isAwake]);

  const speakText = useCallback((text, onEnd) => {
    if (!('speechSynthesis' in window)) {
      onEnd?.();
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice =
      voices.find((v) => /en-US|en_US/i.test(v.lang) && /female|Zira|Samantha|Google US/i.test(v.name)) ||
      voices.find((v) => /en/i.test(v.lang));
    if (preferredVoice) utterance.voice = preferredVoice;
    utterance.rate = 1;
    utterance.pitch = 1.1;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      onEnd?.();
    };
    utterance.onerror = () => {
      setIsSpeaking(false);
      onEnd?.();
    };
    window.speechSynthesis.speak(utterance);
  }, []);

  // Pulls the next pending greeting off the queue (if nobody's currently
  // being greeted) and speaks it. Recurses via speakText's onEnd once the
  // hold period passes, so people are greeted strictly one at a time no
  // matter how many are queued up.
  const processGreetQueue = useCallback(() => {
    if (activeGreetRef.current) return;
    const next = greetQueueRef.current.shift();
    if (!next) return;

    activeGreetRef.current = next;
    lastGreetedAtRef.current.set(next.key, Date.now());
    setActiveGreeting({ name: next.name, title: next.title });

    const text = next.name
      ? next.title
        ? `Welcome back, ${next.title} ${next.name}!`
        : `Welcome back, ${next.name}!`
      : DEFAULT_GREETING;

    speakText(text, () => {
      setTimeout(() => {
        queuedKeysRef.current.delete(next.key);
        activeGreetRef.current = null;
        setActiveGreeting(null);
        processGreetQueue();
      }, GREET_DISPLAY_HOLD_MS);
    });
  }, [speakText]);

  const checkMotion = useCallback((video) => {
    const canvas = canvasRef.current;
    const w = 64;
    const h = 48;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, w, h);
    const frame = ctx.getImageData(0, 0, w, h).data;

    if (!prevFrameRef.current) {
      prevFrameRef.current = frame;
      return 0;
    }
    let diffSum = 0;
    const prev = prevFrameRef.current;
    for (let i = 0; i < frame.length; i += 4) {
      diffSum += Math.abs(frame[i] - prev[i]);
    }
    prevFrameRef.current = frame;
    return diffSum / (w * h);
  }, []);

  const detectFace = useCallback(async () => {
    if (
      modelRef.current &&
      webcamRef.current &&
      webcamRef.current.video &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const predictions = await modelRef.current.estimateFaces(video, false);
      const motionDiff = checkMotion(video);
      const motion = motionDiff > MOTION_THRESHOLD;

      if (predictions.length > 0) {
        const [x1, y1] = predictions[0].topLeft;
        const [x2, y2] = predictions[0].bottomRight;
        const faceCenterX = (x1 + x2) / 2;
        const faceCenterY = (y1 + y2) / 2;

        const nx = (faceCenterX / video.videoWidth - 0.5) * 2;
        const ny = (faceCenterY / video.videoHeight - 0.5) * 2;

        const clampedX = Math.max(-1, Math.min(1, nx)) * (INVERT_X ? -1 : 1);
        const clampedY = Math.max(-1, Math.min(1, ny)) * (INVERT_Y ? -1 : 1);

        setEyeOffset({
          x: clampedX * MAX_PUPIL_OFFSET_X,
          y: clampedY * MAX_PUPIL_OFFSET_Y,
        });
      }

      if (predictions.length > 0 || motion) {
        missCountRef.current = 0;
        setIsFaceDetected(true);
      } else {
        missCountRef.current += 1;
        if (missCountRef.current >= MISS_TOLERANCE) {
          setIsFaceDetected(false);
          setEyeOffset({ x: 0, y: 0 });
          // Scene is empty -- clear the queue and in-progress state so the
          // next person to arrive gets a clean slate. Cooldown timestamps
          // are intentionally kept so a person stepping out and back in
          // within RE_GREET_COOLDOWN_MS isn't re-greeted right away.
          greetQueueRef.current = [];
          queuedKeysRef.current = new Set();
          candidateStreakRef.current = new Map();
          activeGreetRef.current = null;
          setActiveGreeting(null);
        }
      }
    }
  }, [checkMotion]);

  useEffect(() => {
    if (!isModelLoading && hasStarted && introFinished) {
      intervalRef.current = setInterval(detectFace, 300);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isModelLoading, hasStarted, introFinished, detectFace]);

  // Fired by the "Get Started" button. Calling speakText directly inside the
  // click handler (rather than in a useEffect after state changes) keeps it
  // inside the user-gesture context some browsers require for audio.
  // Detection only starts once the intro utterance's onEnd fires, so an
  // early visitor can't cut Trix off mid-introduction.
  const handleGetStarted = useCallback(() => {
    setHasStarted(true);
    speakText(INTRO_GREETING, () => setIntroFinished(true));
  }, [speakText]);

  // Scans every face currently in frame (not just the closest one) and
  // queues up a greeting for each newly-confirmed identity. Runs on its own
  // (slower) interval since computing descriptors is heavier than the
  // BlazeFace presence check above.
  const scanFaces = useCallback(async () => {
    if (
      isRecognizingRef.current ||
      !webcamRef.current ||
      !webcamRef.current.video ||
      webcamRef.current.video.readyState !== 4
    ) {
      return;
    }
    isRecognizingRef.current = true;
    try {
      const faces = await getAllFaceDescriptors(webcamRef.current.video);
      if (faces.length === 0) return;

      const now = Date.now();
      const seenThisScan = new Set();

      for (const face of faces) {
        const match = knownFaces.length > 0 ? matchFace(face.descriptor, knownFaces) : null;
        const key = match ? `${match.name}::${match.title ?? ''}` : UNKNOWN_KEY;
        seenThisScan.add(key);

        const entry = candidateStreakRef.current.get(key) || { score: 0, misses: 0 };
        entry.score += 1;
        entry.misses = 0;
        candidateStreakRef.current.set(key, entry);

        const requiredHits = match
          ? RECOGNITION_CONFIRM_HITS_KNOWN
          : RECOGNITION_CONFIRM_HITS_UNKNOWN;
        const confirmed = entry.score >= requiredHits;
        const alreadyPending = queuedKeysRef.current.has(key);
        const cooldownElapsed =
          now - (lastGreetedAtRef.current.get(key) || 0) > RE_GREET_COOLDOWN_MS;

        if (confirmed && cooldownElapsed && !alreadyPending) {
          greetQueueRef.current.push({
            key,
            name: match ? match.name : null,
            title: match ? match.title : null,
          });
          queuedKeysRef.current.add(key);
          console.log(
            match
              ? `Queued greeting for "${match.name}" (${match.title || 'no title'})`
              : 'Queued greeting for an unrecognized visitor.'
          );
        }
      }

      // Anyone not seen this scan gets docked instead of instantly wiped --
      // tolerates a dropped frame without throwing away real progress. Only
      // drop a candidate once it's been missed repeatedly in a row (or its
      // score has fully decayed), meaning they've actually left frame.
      for (const [key, entry] of candidateStreakRef.current.entries()) {
        if (seenThisScan.has(key)) continue;
        entry.misses += 1;
        entry.score = Math.max(0, entry.score - RECOGNITION_MISS_PENALTY);
        if (entry.misses >= RECOGNITION_DROP_AFTER_MISSES || entry.score <= 0) {
          candidateStreakRef.current.delete(key);
        }
      }

      processGreetQueue();
    } finally {
      isRecognizingRef.current = false;
    }
  }, [knownFaces, processGreetQueue]);

  useEffect(() => {
    if (!isFaceDetected) return;
    // Fire immediately (rather than waiting for the first interval tick) so
    // scanning starts the moment a face shows up.
    scanFaces();
    recognitionIntervalRef.current = setInterval(scanFaces, RECOGNITION_INTERVAL_MS);
    return () => clearInterval(recognitionIntervalRef.current);
  }, [isFaceDetected, scanFaces]);

  const currentState = isModelLoading ? 'loading' : isAwake ? 'happy' : 'sleeping';

  const renderMessage = () => {
    if (isModelLoading) return 'Waking up brain...';
    if (isFaceDetected) {
      if (activeGreeting) {
        if (activeGreeting.name) {
          const greetName = activeGreeting.title
            ? `${activeGreeting.title} ${activeGreeting.name}`
            : activeGreeting.name;
          return `Welcome back, ${greetName}!`;
        }
        return DEFAULT_GREETING;
      }
      return 'Welcome, human!';
    }
    if (isSpeaking) return INTRO_GREETING;
    if (isAwake) return 'Still here...';
    return 'Zzz...';
  };

  if (!hasStarted) {
    return (
      <div className="app-container">
        <div className="start-screen">
          <div className="start-glow" />
          <h1 className="start-title">{ROBOT_NAME}</h1>
          <p className="start-subtitle">
            {isModelLoading ? 'Waking up my brain...' : 'Your friendly robot is ready to meet you.'}
          </p>
          <button
            className="start-button"
            disabled={isModelLoading}
            onClick={handleGetStarted}
          >
            {isModelLoading ? 'Loading...' : 'Get Started'}
          </button>
        </div>

        {/* Mounted early so the camera permission prompt and webcam feed are
            ready the moment the user clicks Get Started. */}
        <div className="hidden-webcam">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{ width: 320, height: 240, facingMode: 'user' }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className={`robot-head ${currentState}`}>
        <RobotFace
          state={currentState}
          isSpeaking={isSpeaking}
          eyeOffset={eyeOffset}
          isBlinking={isBlinking}
        />
      </div>

      <div className="robot-message">{renderMessage()}</div>

      {/* Bottom badge showing the currently-greeted person's title/role */}
      {isFaceDetected && activeGreeting?.title && (
        <div className="robot-subtitle">{activeGreeting.title}</div>
      )}

      <div className="status-bar">
        {isModelLoading ? (
          <span>Loading model</span>
        ) : !introFinished ? (
          <span>Saying hello</span>
        ) : (
          <span>{isFaceDetected ? 'Face detected' : 'No face detected'}</span>
        )}
      </div>

      <div className="hidden-webcam">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 320, height: 240, facingMode: 'user' }}
        />
      </div>
    </div>
  );
}

export default App;