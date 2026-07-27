import { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { loadFaceModels, getFaceDescriptor, saveFace } from './lib/faceRecognition';
import './AdminPage.css';

// Preset roles shown in the dropdown. Add/remove as needed -- picking
// "Other" reveals a free-text field for anything not listed here.
const TITLE_OPTIONS = ['Admin', 'Teacher', 'Dean', 'Staff', 'Student', 'Guest', 'Other'];

function AdminPage() {
  const webcamRef = useRef(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [name, setName] = useState('');
  const [title, setTitle] = useState(TITLE_OPTIONS[0]);
  const [customTitle, setCustomTitle] = useState('');
  const [status, setStatus] = useState('idle'); // idle | capturing | saving | success | error
  const [message, setMessage] = useState('');

  useEffect(() => {
    loadFaceModels().then(() => setModelsReady(true));
  }, []);

  const handleEnroll = useCallback(async () => {
    const finalTitle = title === 'Other' ? customTitle.trim() : title;

    if (!name.trim()) {
      setStatus('error');
      setMessage('Enter a name first.');
      return;
    }
    if (!finalTitle) {
      setStatus('error');
      setMessage('Choose or enter a title (e.g. Admin, Teacher, Dean).');
      return;
    }
    if (!webcamRef.current || !webcamRef.current.video) return;

    setStatus('capturing');
    setMessage('Look at the camera...');

    const descriptor = await getFaceDescriptor(webcamRef.current.video);
    if (!descriptor) {
      setStatus('error');
      setMessage('No face detected. Try again with better lighting.');
      return;
    }

    setStatus('saving');
    try {
      await saveFace(name.trim(), finalTitle, descriptor);
      setStatus('success');
      setMessage(`Saved "${name.trim()}" (${finalTitle})!`);
      setName('');
      setCustomTitle('');
      setTitle(TITLE_OPTIONS[0]);
    } catch (err) {
      setStatus('error');
      setMessage(err.message || 'Failed to save face.');
    }
  }, [name, title, customTitle]);

  const isBusy = status === 'capturing' || status === 'saving';

  return (
    <div className="admin-container">
      <h1>Enroll a New Face</h1>
      <p className="admin-subtitle">
        {modelsReady ? 'Models loaded. Enter details, then capture.' : 'Loading recognition models...'}
      </p>

      <div className="admin-webcam-wrap">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 320, height: 240, facingMode: 'user' }}
        />
      </div>

      <input
        type="text"
        placeholder="Person's name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="admin-input"
        disabled={isBusy}
      />

      <select
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        className="admin-select"
        disabled={isBusy}
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
          className="admin-input"
          disabled={isBusy}
        />
      )}

      <button
        onClick={handleEnroll}
        disabled={!modelsReady || isBusy}
        className="admin-button"
      >
        {isBusy ? 'Working...' : 'Capture & Save'}
      </button>

      {message && <p className={`admin-message ${status}`}>{message}</p>}

      <a className="admin-back-link" href="/">
        &larr; Back to robot
      </a>
    </div>
  );
}

export default AdminPage;