// Talks to Groq's free, OpenAI-compatible chat endpoint.
// Get a free key (no credit card) at https://console.groq.com -> API Keys,
// then put it in your .env as VITE_GROQ_API_KEY=gsk_...

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

// Kept short/plain on purpose -- this gets read aloud by speechSynthesis,
// so long paragraphs or markdown/bullets would sound bad spoken out loud.
const SYSTEM_PROMPT = `You are Trix, a friendly robot assistant working the front desk,
similar to a helpful customer support agent. You answer questions naturally and
conversationally. Keep every answer SHORT -- 1 to 3 sentences, spoken out loud by
text-to-speech, so no lists, no markdown, no headings, just plain natural sentences.
If you don't know something specific about this location/company, say so briefly and
offer to help some other way instead of making facts up.`;

// Quick, deterministic answer for "who made/developed you" questions -- this
// skips the LLM entirely so credits are always exact, never paraphrased or
// hallucinated by the model.
const CREATOR_QUESTION_REGEX =
  /\b(who\s+(made|built|created|developed|designed|programmed)\s+you|who('?s| is)\s+your\s+(developer|creator|maker|programmer)|who\s+(is|are)\s+(the\s+)?(developer|creator|programmer|program\s*head)s?|(created|developed|built)\s+you)\b/i;

const CREATOR_ANSWER =
  "I was developed by Carlo Cañezares. Keith Andri Mag-aso provided emotional support, and Procoro Gonzaga is the program head, and Team OJT's";

// Quick, deterministic answer for "what time/date is it" questions -- the
// LLM has no access to the real clock and was guessing/hallucinating a time,
// so this skips it entirely and answers with the actual current time in the
// Philippines (Asia/Manila), regardless of what timezone the server/browser
// itself happens to be running in.
const TIME_QUESTION_REGEX =
  /\bwhat(?:'?s| is)?\s+(?:the\s+)?time(?:\s+(?:is\s+it|right\s+now|now))?\b|\bcurrent\s+time\b|\btime\s+(?:is\s+it|now)\b/i;

const DATE_QUESTION_REGEX =
  /\bwhat(?:'?s| is)?\s+(?:the\s+)?date\b|\bwhat\s+day\s+is\s+it(?:\s+today)?\b|\btoday'?s?\s+date\b|\bwhat'?s?\s+today'?s?\s+date\b/i;

const MANILA_TIME_ZONE = 'Asia/Manila';

const getManilaTimeAnswer = () => {
  const time = new Date().toLocaleTimeString('en-US', {
    timeZone: MANILA_TIME_ZONE,
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
  return `It's ${time} right now.`;
};

const getManilaDateAnswer = () => {
  const date = new Date().toLocaleDateString('en-US', {
    timeZone: MANILA_TIME_ZONE,
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  return `Today is ${date}.`;
};

// conversationHistory: array of { role: 'user' | 'assistant', content: string }
// personName: optional, lets Trix address a recognized visitor by name.
export async function askAI(question, conversationHistory = [], personName = null) {
  if (CREATOR_QUESTION_REGEX.test(question)) {
    return CREATOR_ANSWER;
  }
  if (TIME_QUESTION_REGEX.test(question)) {
    return getManilaTimeAnswer();
  }
  if (DATE_QUESTION_REGEX.test(question)) {
    return getManilaDateAnswer();
  }

  const apiKey = import.meta.env.VITE_GROQ_API_KEY;
  if (!apiKey) {
    throw new Error('Missing VITE_GROQ_API_KEY -- add it to your .env file.');
  }

  const systemContent = personName
    ? `${SYSTEM_PROMPT} You are currently speaking with ${personName}; you may address them by name.`
    : SYSTEM_PROMPT;

  const messages = [
    { role: 'system', content: systemContent },
    ...conversationHistory,
    { role: 'user', content: question },
  ];

  const response = await fetch(GROQ_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages,
      temperature: 0.6,
      max_tokens: 200,
    }),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => '');
    throw new Error(`Groq API error ${response.status}: ${errText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content?.trim() || "Sorry, I didn't catch that.";
}