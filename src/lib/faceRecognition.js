import * as faceapi from '@vladmandic/face-api';
import { supabase } from './supabaseClient';

// How close (Euclidean distance) two descriptors must be to count as the
// same person. face-api.js descriptors are 128-d; ~0.5-0.6 is a common
// threshold. Lower = stricter match.
export const MATCH_THRESHOLD = 0.5;

// If the second-closest known face is within this distance of the best
// match, we treat the result as ambiguous and report "no match" rather than
// guess. This matters most with multiple people in frame or lookalike
// enrollments -- it's better to say "Welcome, visitor!" than to greet the
// wrong person confidently.
export const MATCH_MARGIN = 0.08;

let modelsLoaded = false;

// A smaller detector input size runs noticeably faster per frame at the
// cost of some range/accuracy on faces far from the camera -- a good
// trade-off for a robot meant to recognize people up close in real time.
// Reused across calls instead of constructing a new options object each
// scan.
const DETECTOR_OPTIONS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5,
});

// Models are loaded from /public/models (see setup notes for downloading them)
export async function loadFaceModels() {
  if (modelsLoaded) return;
  const MODEL_URL = '/models';
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  modelsLoaded = true;
}

// Detects every face in the frame and returns their descriptors sorted
// largest (closest to camera) first. Used by the main app to track and
// queue up everyone in view, instead of reacting to only one face.
export async function getAllFaceDescriptors(videoOrImage) {
  const detections = await faceapi
    .detectAllFaces(videoOrImage, DETECTOR_OPTIONS)
    .withFaceLandmarks()
    .withFaceDescriptors();

  if (!detections || detections.length === 0) return [];

  return detections
    .map((d) => ({
      descriptor: d.descriptor,
      area: d.detection.box.width * d.detection.box.height,
    }))
    .sort((a, b) => b.area - a.area);
}

// Returns a Float32Array(128) descriptor for the largest (closest) face in
// the video/image, or null if no face was found. Used by the admin
// enrollment page, where only one person is expected in frame at a time.
export async function getFaceDescriptor(videoOrImage) {
  const faces = await getAllFaceDescriptors(videoOrImage);
  return faces.length > 0 ? faces[0].descriptor : null;
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

// Fetch all known faces once; call again if you enroll someone new mid-session.
export async function fetchKnownFaces() {
  const { data, error } = await supabase
    .from('known_faces')
    .select('id, name, title, embedding');
  if (error) {
    console.error('Failed to fetch known faces:', error);
    return [];
  }
  // Supabase returns the pgvector column as a string like "[0.1,0.2,...]"
  return data.map((row) => ({
    id: row.id,
    name: row.name,
    title: row.title || null,
    embedding: Array.isArray(row.embedding) ? row.embedding : JSON.parse(row.embedding),
  }));
}

// Compares a live descriptor against the cached known faces (client-side
// matching -- fine for a personal robot with a modest number of faces).
export function matchFace(liveDescriptor, knownFaces) {
  let best = null;
  let bestDistance = Infinity;
  let secondDistance = Infinity;

  for (const face of knownFaces) {
    const distance = euclideanDistance(liveDescriptor, face.embedding);
    if (distance < bestDistance) {
      secondDistance = bestDistance;
      bestDistance = distance;
      best = face;
    } else if (distance < secondDistance) {
      secondDistance = distance;
    }
  }

  if (!best || bestDistance >= MATCH_THRESHOLD) return null;

  // Someone else in the known-faces list is nearly as close a match --
  // too ambiguous to confidently say who this is.
  if (secondDistance - bestDistance < MATCH_MARGIN) return null;

  return { name: best.name, title: best.title, distance: bestDistance };
}

// Used by the admin page to save a newly enrolled face.
export async function saveFace(name, title, descriptor) {
  const { error } = await supabase
    .from('known_faces')
    .insert({ name, title, embedding: Array.from(descriptor) });
  if (error) throw error;
}

// --- Admin dashboard CRUD ---------------------------------------------
// The functions below back the "manage registered users" table on the
// admin page. They intentionally never select/return the `embedding`
// column -- it's a 128-length float array per row and the dashboard has
// no use for it, so leaving it out keeps list fetches light.

// Lists every enrolled person, most recently added first. Falls back to
// ordering by id if the table has no created_at column.
export async function fetchAllFaces() {
  let { data, error } = await supabase
    .from('known_faces')
    .select('id, name, title, created_at')
    .order('created_at', { ascending: false });

  if (error) {
    // created_at may not exist on older schemas -- retry without it.
    ({ data, error } = await supabase
      .from('known_faces')
      .select('id, name, title')
      .order('id', { ascending: false }));
  }

  if (error) {
    console.error('Failed to fetch faces:', error);
    throw error;
  }
  return data;
}

// Updates a person's name and/or title. Pass only the fields you want to
// change, e.g. updateFace(id, { title: 'Teacher' }).
export async function updateFace(id, updates) {
  const { error } = await supabase
    .from('known_faces')
    .update(updates)
    .eq('id', id);
  if (error) throw error;
}

// Deletes an enrolled person (and their face embedding) entirely.
export async function deleteFace(id) {
  const { error } = await supabase
    .from('known_faces')
    .delete()
    .eq('id', id);
  if (error) throw error;
}


/// FILE: supabaseClient.js ///