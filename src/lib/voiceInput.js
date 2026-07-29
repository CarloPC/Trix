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
export function listenOnce({ onStart, onInterim, onResult, onError, onEnd, timeoutMs = 8000 }) {
  if (!SpeechRecognitionImpl) {
    onError?.(new Error('Speech recognition is not supported in this browser.'));
    onEnd?.();
    return () => {};
  }

  const recognition = new SpeechRecognitionImpl();
  recognition.lang = 'en-US';
  // interimResults=true is what lets us stream words to the screen as the
  // person talks, instead of only showing text once they stop -- the
  // "real-time transcript" behavior of Gemini/ChatGPT/etc.
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;

  let settled = false;
  const timeout = setTimeout(() => {
    if (!settled) recognition.stop(); // no speech detected in time
  }, timeoutMs);

  recognition.onstart = () => onStart?.();

  recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
    }
    if (interimTranscript) onInterim?.(interimTranscript);
    if (finalTranscript) {
      settled = true;
      clearTimeout(timeout);
      onResult?.(finalTranscript.trim());
    }
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

  // Chrome throws a synchronous "InvalidStateError" ("already started") if
  // start() is called while a *previous* SpeechRecognition session (e.g.
  // the wake-word listener that was just stopped to hand off to this one)
  // hasn't fully torn down yet -- stop() is async, so there's a real
  // window where that race happens. When start() throws, onresult/onend
  // never fire at all, so the question is silently dropped and the robot
  // just looks like it "isn't responding". Retrying a beat later covers
  // the handoff race instead of failing outright.
  const attemptStart = (retriesLeft = 4) => {
    try {
      recognition.start();
    } catch (err) {
      if (retriesLeft > 0) {
        setTimeout(() => attemptStart(retriesLeft - 1), 150);
      } else {
        onError?.(err);
        onEnd?.();
      }
    }
  };
  attemptStart();

  return () => {
    settled = true;
    clearTimeout(timeout);
    recognition.stop();
  };
}

// Continuously listens in the background for one of the given wake phrases
// (case-insensitive substring match -- "trix", "hi trix", "hello trix" all
// match on the word "trix" alone). Auto-restarts itself if the browser times
// out an idle continuous session, so the robot can be "always listening" for
// its name. Returns stop() -- call it before starting a one-shot listenOnce(),
// since only one SpeechRecognition session should run at a time.
// Covers common speech-to-text mishearings of "Trix" -- it's a short,
// unusual word, and Chrome's recognizer frequently hears "tricks", "trick",
// or "trish" instead. Without these, "instantly listen" can silently fail
// just because the engine transcribed the name slightly wrong.
const DEFAULT_WAKE_WORDS = ['trix', 'tricks', 'trick', 'trish', 'tris'];

export function listenForWakeWord({ wakeWords = DEFAULT_WAKE_WORDS, onWake, onError }) {
  if (!SpeechRecognitionImpl) {
    onError?.(new Error('Speech recognition is not supported in this browser.'));
    return () => {};
  }

  const normalizedWakeWords = wakeWords.map((w) => w.toLowerCase());
  const containsWakeWord = (text) => {
    const lower = text.toLowerCase();
    return normalizedWakeWords.some((w) => lower.includes(w));
  };

  let stopped = false;
  let woke = false;
  let recognition = null;

  const start = () => {
    if (stopped) return;
    recognition = new SpeechRecognitionImpl();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = true;

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (containsWakeWord(transcript)) {
          // Don't fire onWake yet -- stop() is async and the mic session
          // isn't actually released until onend fires below. Starting a
          // new SpeechRecognition (for the follow-up question) before that
          // happens throws "already started" in Chrome, which was silently
          // swallowing every question asked right after the wake word.
          stopped = true;
          woke = true;
          recognition.stop();
          return;
        }
      }
    };

    recognition.onerror = (event) => {
      // "no-speech" / "aborted" happen constantly on a long-running
      // listener -- only surface real problems (e.g. mic permission denied).
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        onError?.(new Error(event.error));
      }
    };

    recognition.onend = () => {
      // Browsers silently end long "continuous" sessions after a while --
      // restart automatically unless we've been told to stop.
      if (woke) {
        woke = false;
        onWake?.(); // mic is now actually free -- safe to start listenOnce()
        return;
      }
      if (!stopped) start();
    };

    try {
      recognition.start();
    } catch {
      // Transient "already started" error. onend won't fire for a session
      // that never actually started, so without this retry the listener
      // could get permanently stuck instead of recovering on its own.
      if (!stopped) setTimeout(start, 200);
    }
  };

  start();

  return () => {
    stopped = true;
    recognition?.stop();
  };
}