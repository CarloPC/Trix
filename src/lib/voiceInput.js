// Wraps the browser's built-in SpeechRecognition (free, no API key, no network
// call to us) so the robot can listen to a spoken question and hand back text.
// Supported in Chrome/Edge; not supported in Firefox and older Safari -- callers
// should check isSpeechRecognitionSupported() and offer a text-input fallback.

const SpeechRecognitionImpl =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

export function isSpeechRecognitionSupported() {
  return !!SpeechRecognitionImpl;
}

// Listens for a single utterance. Calls onResult(transcript) when the person
// finishes speaking, onError(err) on failure/no-speech/permission-denied, and
// always calls onEnd() when the mic closes (success or failure). Returns a
// cancel() function you can call to stop listening early.
export function listenOnce({ onStart, onResult, onError, onEnd, timeoutMs = 8000 }) {
  if (!SpeechRecognitionImpl) {
    onError?.(new Error('Speech recognition is not supported in this browser.'));
    onEnd?.();
    return () => {};
  }

  const recognition = new SpeechRecognitionImpl();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;

  let settled = false;
  const timeout = setTimeout(() => {
    if (!settled) recognition.stop(); // no speech detected in time
  }, timeoutMs);

  recognition.onstart = () => onStart?.();

  recognition.onresult = (event) => {
    settled = true;
    clearTimeout(timeout);
    const transcript = event.results[0][0].transcript;
    onResult?.(transcript);
  };

  recognition.onerror = (event) => {
    settled = true;
    clearTimeout(timeout);
    onError?.(new Error(event.error || 'speech-recognition-error'));
  };

  recognition.onend = () => {
    clearTimeout(timeout);
    onEnd?.();
  };

  try {
    recognition.start();
  } catch (err) {
    onError?.(err);
    onEnd?.();
  }

  return () => {
    settled = true;
    clearTimeout(timeout);
    recognition.stop();
  };
}