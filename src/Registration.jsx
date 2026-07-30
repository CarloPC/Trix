import { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { loadFaceModels, getFaceDescriptor, saveFace } from './lib/faceRecognition';
import './Registration.css';

// Preset roles shown in the dropdown. Add/remove as needed -- picking
// "Other" reveals a free-text field for anything not listed here.
const TITLE_OPTIONS = ['Admin', 'Teacher', 'Dean', 'Staff', 'Student', 'Guest', 'Other'];

// After a successful save, how long to show the confirmation before sending
// the person back to Trix's main screen (not the admin dashboard).
const SUCCESS_REDIRECT_MS = 1800;
const ROBOT_HOME_PATH = '/';

function Registration() {
  const webcamRef = useRef(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [name, setName] = useState('');
  const [title, setTitle] = useState(TITLE_OPTIONS[0]);
  const [customTitle, setCustomTitle] = useState('');
  // idle -> live camera, entering details
  // capturing -> pulling a descriptor from the current frame
  // review -> a frame's been captured; waiting on confirm/retake
  // saving -> writing the confirmed capture to Supabase
  // success -> saved; about to redirect back to the robot
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('');
  // Kept separate from `status` so an error while saving a *confirmed*
  // capture can be shown without losing the review screen (retake/confirm
  // stay available instead of being bounced back to the live camera).
  const [messageType, setMessageType] = useState('info'); // info | error | success
  // The exact frame the descriptor was pulled from, so the person can
  // actually see/approve what's about to be saved.
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  // Holds the Float32Array descriptor between capture and confirm -- a ref
  // since it doesn't need to trigger re-renders itself.
  const capturedDescriptorRef = useRef(null);
  const redirectTimeoutRef = useRef(null);

  useEffect(() => {
    loadFaceModels().then(() => setModelsReady(true));
  }, []);

  useEffect(() => {
    return () => {
      if (redirectTimeoutRef.current) clearTimeout(redirectTimeoutRef.current);
    };
  }, []);

  // Step 1: pull a descriptor from the live camera and freeze a preview of
  // that same frame. Doesn't save anything yet -- that only happens once
  // the person confirms the preview looks right.
  const handleCapture = useCallback(async () => {
    const finalTitle = title === 'Other' ? customTitle.trim() : title;

    if (!name.trim()) {
      setMessageType('error');
      setMessage('Enter a name first.');
      return;
    }
    if (!finalTitle) {
      setMessageType('error');
      setMessage('Choose or enter a title (e.g. Admin, Teacher, Dean).');
      return;
    }
    if (!webcamRef.current || !webcamRef.current.video) return;

    setStatus('capturing');
    setMessageType('info');
    setMessage('Look at the camera...');

    const descriptor = await getFaceDescriptor(webcamRef.current.video);
    if (!descriptor) {
      setStatus('idle');
      setMessageType('error');
      setMessage('No face detected. Try again with better lighting.');
      return;
    }

    const screenshot = webcamRef.current.getScreenshot();
    capturedDescriptorRef.current = descriptor;
    setCapturedPhoto(screenshot);
    setStatus('review');
    setMessageType('info');
    setMessage('Does this photo look good?');
  }, [name, title, customTitle]);

  // Step 2a: person didn't like the preview -- discard it and go back to
  // the live camera so they can try again.
  const handleRetake = useCallback(() => {
    capturedDescriptorRef.current = null;
    setCapturedPhoto(null);
    setStatus('idle');
    setMessageType('info');
    setMessage('');
  }, []);

  // Step 2b: person approved the preview -- now (and only now) actually
  // save the face, then hand off to the robot's main screen.
  const handleConfirmSave = useCallback(async () => {
    const finalTitle = title === 'Other' ? customTitle.trim() : title;
    if (!capturedDescriptorRef.current) {
      // Guards against a stray click with nothing captured -- shouldn't be
      // reachable since this button only renders during 'review'.
      setStatus('idle');
      return;
    }

    setStatus('saving');
    setMessageType('info');
    setMessage('Saving...');

    try {
      await saveFace(name.trim(), finalTitle, capturedDescriptorRef.current);
      setStatus('success');
      setMessageType('success');
      setMessage(`Saved "${name.trim()}" (${finalTitle})! Heading back to Trix...`);
      redirectTimeoutRef.current = setTimeout(() => {
        window.location.href = ROBOT_HOME_PATH;
      }, SUCCESS_REDIRECT_MS);
    } catch (err) {
      // Stay on the review screen (photo + confirm/retake) rather than
      // bouncing back to the live camera -- the capture itself was fine,
      // only the save failed, so there's no reason to make them retake it.
      setStatus('review');
      setMessageType('error');
      setMessage(err.message || 'Failed to save face. Try again.');
    }
  }, [name, title, customTitle]);

  const isBusy = status === 'capturing' || status === 'saving';
  const isReviewing = status === 'review' || status === 'saving';

  return (
    <div className="registration-container">
      <h1>Enroll a New Face</h1>
      <p className="registration-subtitle">
        {modelsReady ? 'Models loaded. Enter details, then capture.' : 'Loading recognition models...'}
      </p>

      <div className="registration-webcam-wrap">
        {isReviewing && capturedPhoto ? (
          <img src={capturedPhoto} alt="Captured preview" className="registration-preview-photo" />
        ) : (
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{ width: 320, height: 240, facingMode: 'user' }}
          />
        )}
      </div>

      <input
        type="text"
        placeholder="Person's name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="registration-input"
        disabled={isBusy || isReviewing}
      />

      <select
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        className="registration-select"
        disabled={isBusy || isReviewing}
      >
        {TITLE_OPTIONS.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>

      {title === 'Other' && (
        <input
          type="text"
          placeholder="Enter custom title"
          value={customTitle}
          onChange={(e) => setCustomTitle(e.target.value)}
          className="registration-input"
          disabled={isBusy || isReviewing}
        />
      )}

      {status === 'review' || status === 'saving' ? (
        <div className="registration-review-actions">
          <button
            onClick={handleRetake}
            disabled={status === 'saving'}
            className="registration-button registration-retake-button"
          >
            Retake
          </button>
          <button
            onClick={handleConfirmSave}
            disabled={status === 'saving'}
            className="registration-button"
          >
            {status === 'saving' ? 'Saving...' : 'Confirm & Save'}
          </button>
        </div>
      ) : (
        <button
          onClick={handleCapture}
          disabled={!modelsReady || isBusy}
          className="registration-button"
        >
          {status === 'capturing' ? 'Working...' : 'Capture'}
        </button>
      )}

      {message && <p className={`registration-message ${messageType}`}>{message}</p>}

      <a className="registration-back-link" href="/">
        &larr; Back to Trix
      </a>
    </div>
  );
}

export default Registration;