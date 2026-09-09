import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";

const PLAYBACK_FPS = 30;  // ✅ IMPROVED: Increased from 20 for smoother playback
let lastFrameTime = 0;
const SMOOTHING = 0.35;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const stageEl = document.getElementById("avatarStage");
const initialW = stageEl?.clientWidth || window.innerWidth;
const initialH = stageEl?.clientHeight || window.innerHeight;
const camera = new THREE.PerspectiveCamera(35, initialW / initialH, 0.01, 100);
camera.position.set(0, 1.2, 0.95);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(stageEl?.clientWidth || window.innerWidth, stageEl?.clientHeight || window.innerHeight);
if (stageEl) {
  stageEl.appendChild(renderer.domElement);
} else {
  document.body.appendChild(renderer.domElement);
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.2, 0);
controls.update();

function frameAvatarUpperBody() {
  if (!avatar) return;
  const box = new THREE.Box3().setFromObject(avatar);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  // Focus around chest/shoulder zone so legs are naturally cropped.
  const focusY = center.y + size.y * 0.04;
  const focus = new THREE.Vector3(center.x, focusY, center.z);

  // Keep distance relative to model height so framing stays consistent
  // across different avatar assets.
  const distance = Math.max(0.95, size.y * 0.58);
  camera.position.set(focus.x, focus.y + size.y * 0.06, focus.z + distance);
  controls.target.copy(focus);
  controls.update();
}

function normalizeAvatarSize(targetHeight = 1.9) {
  if (!avatar) return;
  const box = new THREE.Box3().setFromObject(avatar);
  if (box.isEmpty()) return;
  const h = box.getSize(new THREE.Vector3()).y;
  if (!h || h < 1e-6) return;
  const k = targetHeight / h;
  avatar.scale.multiplyScalar(k);
}

scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
scene.add(new THREE.DirectionalLight(0xffffff, 0.6));

// Lifts pose + hand skeleton to match GLB (feet/root vs normalized landmarks); tune if overlay drifts
const MP_ALIGN_Y = 0.52;

// ---------- GLTF avatar ----------
// Path is relative to project root served by http.server
// Put a rigged GLB at: frontend/avatar_viewer/models/avatar.glb
let avatar = null;
let bonesByName = {};
let avatarReady = false;

const loader = new GLTFLoader();
loader.load(
  "/frontend/avatar_viewer/models/avatar.glb",
  (gltf) => {
    avatar = gltf.scene;
    avatar.traverse((obj) => {
      if (obj.isBone) {
        bonesByName[obj.name] = obj;
      }
    });
    // Same vertical space as mpToWorld() + MP_ALIGN_Y so mesh lines up with stick figure
    avatar.position.set(0, MP_ALIGN_Y - 0.05, 0);
    avatar.scale.set(1, 1, 1);
    normalizeAvatarSize(1.9);

    // Try to automatically guess common arm bone names if user did not edit POSE_TO_BONE
    const boneNames = Object.keys(bonesByName);
    console.log("Avatar bones:", boneNames);

    scene.add(avatar);
    avatarReady = true;
    frameAvatarUpperBody();
  },
  undefined,
  (err) => {
    console.warn("Avatar GLB failed to load:", err);
  }
);

// ---------- helpers ----------
function makeLine(color = 0xffffff) {
  const geom = new THREE.BufferGeometry();
  const pos = new Float32Array(2 * 3);
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.LineBasicMaterial({ color });
  const line = new THREE.Line(geom, mat);
  line.frustumCulled = false;
  return line;
}

function setLine(line, a, b) {
  const arr = line.geometry.attributes.position.array;
  arr[0] = a[0]; arr[1] = a[1]; arr[2] = a[2];
  arr[3] = b[0]; arr[4] = b[1]; arr[5] = b[2];
  line.geometry.attributes.position.needsUpdate = true;
}

function makeJoint(radius = 0.028, color = 0xffccaa) {
  const geo = new THREE.SphereGeometry(radius, 14, 14);
  const mat = new THREE.MeshStandardMaterial({ color });
  const m = new THREE.Mesh(geo, mat);
  m.frustumCulled = false;
  scene.add(m);
  return m;
}

function mpToWorld(lm) {
  if (!lm || lm.length < 3) return [0, 0, 0];
  const x = (lm[0] - 0.5) * 1.35;
  const y = (0.85 - lm[1]) * 1.35 + MP_ALIGN_Y;
  const z = (-lm[2]) * 0.75;
  return [x, y, z];
}

function smoothLandmarks(prevArr, currArr) {
  if (!prevArr) return currArr;
  if (!currArr) return prevArr;

  const out = [];
  for (let i = 0; i < currArr.length; i++) {
    const p = prevArr[i] || [0, 0, 0];
    const c = currArr[i] || [0, 0, 0];
    out.push([
      p[0] + (c[0] - p[0]) * SMOOTHING,
      p[1] + (c[1] - p[1]) * SMOOTHING,
      p[2] + (c[2] - p[2]) * SMOOTHING,
    ]);
  }
  return out;
}

// ✅ Detect if hand landmarks look valid (avoid fan/spikes)
function handLooksValid(hand) {
  if (!hand || hand.length < 21) return false;

  // count non-zero points
  let nonZero = 0;
  let minX =  999, minY =  999;
  let maxX = -999, maxY = -999;

  for (let i = 0; i < hand.length; i++) {
    const [x, y, z] = hand[i] || [0, 0, 0];
    if (Math.abs(x) > 1e-4 || Math.abs(y) > 1e-4 || Math.abs(z) > 1e-4) {
      nonZero++;
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
  }

  // if too many zeros => missing hand
  if (nonZero < 12) return false;

  // if hand bounding box is tiny => likely collapsed
  const spread = (maxX - minX) + (maxY - minY);
  if (spread < 0.03) return false;

  // if values are crazy (sometimes bad frames jump out)
  if (minX < -0.2 || minY < -0.2 || maxX > 1.2 || maxY > 1.2) return false;

  return true;
}

// ---------- skeleton topology ----------
const HAND_EDGES = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
];

const POSE_EDGES = [
  [11,12],[11,23],[12,24],[23,24],  // torso
  [11,13],[13,15],                  // left arm
  [12,14],[14,16],                  // right arm
  [11,0],[12,0]                     // neck to nose
];

// ---------- render objects ----------
let bodyLines = [];
let leftHandLines = [];
let rightHandLines = [];
let poseJoints = [];
let headMesh = null;

let torsoMesh = null;
const torsoGeom = new THREE.BufferGeometry();
const torsoMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.08,
  side: THREE.DoubleSide
});

// ---------- avatar retargeting ----------
const POSE_TO_BONE = {
  // MediaPipe 11 (left shoulder-ish) → upper arm bone
  11: "LeftArm",
  // MediaPipe 13 (left elbow) → forearm bone
  13: "LeftForeArm",
  // MediaPipe 15 (left wrist) → hand bone
  15: "LeftHand",

  // MediaPipe 12 (right shoulder-ish) → upper arm bone
  12: "RightArm",
  // MediaPipe 14 (right elbow) → forearm bone
  14: "RightForeArm",
  // MediaPipe 16 (right wrist) → hand bone
  16: "RightHand",
};

const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();

/** Aim bone in parent space so it points from world A → world B (typical +Y rest pose). */
function aimBoneLocal(bone, worldA, worldB) {
  if (!bone || !bone.parent) return;
  bone.parent.updateWorldMatrix(true, false);
  const inv = new THREE.Matrix4().copy(bone.parent.matrixWorld).invert();
  _vA.copy(worldA).applyMatrix4(inv);
  _vB.copy(worldB).applyMatrix4(inv);
  const dir = _vB.sub(_vA);
  if (dir.lengthSq() < 1e-8) return;
  dir.normalize();
  const rest = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(rest, dir);
  bone.quaternion.slerp(q, SMOOTHING);
}

function updateBoneFromJoints(boneName, parentIdx, childIdx, pose) {
  if (!avatarReady) return;
  const bone = bonesByName[boneName];
  if (!bone) return;
  const p0 = new THREE.Vector3(...mpToWorld(pose[parentIdx]));
  const p1 = new THREE.Vector3(...mpToWorld(pose[childIdx]));
  aimBoneLocal(bone, p0, p1);
}

// MediaPipe hand indices → bone name chains (same naming as Mixamo-style GLB)
const HAND_FINGER_CHAINS_LEFT = [
  ["LeftHandThumb1", 0, 1],
  ["LeftHandThumb2", 1, 2],
  ["LeftHandThumb3", 2, 3],
  ["LeftHandThumb4", 3, 4],
  ["LeftHandIndex1", 5, 6],
  ["LeftHandIndex2", 6, 7],
  ["LeftHandIndex3", 7, 8],
  ["LeftHandIndex4", 7, 8],
  ["LeftHandMiddle1", 9, 10],
  ["LeftHandMiddle2", 10, 11],
  ["LeftHandMiddle3", 11, 12],
  ["LeftHandMiddle4", 11, 12],
  ["LeftHandRing1", 13, 14],
  ["LeftHandRing2", 14, 15],
  ["LeftHandRing3", 15, 16],
  ["LeftHandRing4", 15, 16],
  ["LeftHandPinky1", 17, 18],
  ["LeftHandPinky2", 18, 19],
  ["LeftHandPinky3", 19, 20],
  ["LeftHandPinky4", 19, 20],
];

const HAND_FINGER_CHAINS_RIGHT = [
  ["RightHandThumb1", 0, 1],
  ["RightHandThumb2", 1, 2],
  ["RightHandThumb3", 2, 3],
  ["RightHandThumb4", 3, 4],
  ["RightHandIndex1", 5, 6],
  ["RightHandIndex2", 6, 7],
  ["RightHandIndex3", 7, 8],
  ["RightHandIndex4", 7, 8],
  ["RightHandMiddle1", 9, 10],
  ["RightHandMiddle2", 10, 11],
  ["RightHandMiddle3", 11, 12],
  ["RightHandMiddle4", 11, 12],
  ["RightHandRing1", 13, 14],
  ["RightHandRing2", 14, 15],
  ["RightHandRing3", 15, 16],
  ["RightHandRing4", 15, 16],
  ["RightHandPinky1", 17, 18],
  ["RightHandPinky2", 18, 19],
  ["RightHandPinky3", 19, 20],
  ["RightHandPinky4", 19, 20],
];

function updateHandBonesFromLandmarks(hand, chains) {
  if (!avatarReady || !hand || hand.length < 21) return;
  // Orient hand bone: wrist → middle MCP (stable palm forward)
  const leftHand = chains[0][0].startsWith("Left");
  const handBoneName = leftHand ? "LeftHand" : "RightHand";
  const hb = bonesByName[handBoneName];
  if (hb) {
    const w = new THREE.Vector3(...mpToWorld(hand[0]));
    const palm = new THREE.Vector3(...mpToWorld(hand[9]));
    aimBoneLocal(hb, w, palm);
  }
  for (const [boneName, ia, ib] of chains) {
    const b = bonesByName[boneName];
    if (!b) continue;
    const wa = new THREE.Vector3(...mpToWorld(hand[ia]));
    const wb = new THREE.Vector3(...mpToWorld(hand[ib]));
    aimBoneLocal(b, wa, wb);
  }
}

// smoothing caches
let prevPose = null;
let prevLH = null;
let prevRH = null;

function clearAvatar() {
  for (const l of bodyLines) scene.remove(l);
  for (const l of leftHandLines) scene.remove(l);
  for (const l of rightHandLines) scene.remove(l);
  for (const j of poseJoints) scene.remove(j);
  if (headMesh) scene.remove(headMesh);
  if (torsoMesh) scene.remove(torsoMesh);

  bodyLines = [];
  leftHandLines = [];
  rightHandLines = [];
  poseJoints = [];
  headMesh = null;
  torsoMesh = null;

  prevPose = null;
  prevLH = null;
  prevRH = null;
}

function buildAvatarOnce() {
  clearAvatar();

  // body lines
  for (let i = 0; i < POSE_EDGES.length; i++) {
    const l = makeLine(0xffffff);
    bodyLines.push(l);
    scene.add(l);
  }

  // hands lines (white), but we will hide them when invalid
  for (let i = 0; i < HAND_EDGES.length; i++) {
    const l = makeLine(0xffffff);
    leftHandLines.push(l);
    scene.add(l);
  }
  for (let i = 0; i < HAND_EDGES.length; i++) {
    const l = makeLine(0xffffff);
    rightHandLines.push(l);
    scene.add(l);
  }

  // body joints only (no green dots)
  const idxs = [0, 11, 12, 13, 14, 15, 16, 23, 24];
  for (let i = 0; i < idxs.length; i++) {
    poseJoints.push(makeJoint(0.03, 0xffccaa));
  }

  // head (slightly smaller than before)
  headMesh = makeJoint(0.065, 0xffccaa);

  // torso fill
  torsoMesh = new THREE.Mesh(torsoGeom, torsoMat);
  torsoMesh.frustumCulled = false;
  scene.add(torsoMesh);
}

function updateTorsoFill(pose) {
  const ls = mpToWorld(pose[11]);
  const rs = mpToWorld(pose[12]);
  const lh = mpToWorld(pose[23]);
  const rh = mpToWorld(pose[24]);

  const verts = new Float32Array([
    ls[0], ls[1], ls[2],
    rs[0], rs[1], rs[2],
    rh[0], rh[1], rh[2],

    ls[0], ls[1], ls[2],
    rh[0], rh[1], rh[2],
    lh[0], lh[1], lh[2],
  ]);

  torsoGeom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  torsoGeom.computeVertexNormals();
}

function setHandVisibility(which, visible) {
  const arr = which === "L" ? leftHandLines : rightHandLines;
  for (const l of arr) l.visible = visible;
}

function renderFrame(f) {
  if (!f || !f.pose) return;

  // Some clips might store frames as {pose,left_hand,right_hand} but may have missing hands.
  const rawPose = f.pose;
  const rawLH = f.left_hand || null;
  const rawRH = f.right_hand || null;

  if (!rawPose || rawPose.length < 25) return;

  const pose = smoothLandmarks(prevPose, rawPose);
  prevPose = pose;

  const lhValid = handLooksValid(rawLH);
  const rhValid = handLooksValid(rawRH);

  const lh = lhValid ? smoothLandmarks(prevLH, rawLH) : null;
  const rh = rhValid ? smoothLandmarks(prevRH, rawRH) : null;

  prevLH = lhValid ? lh : prevLH;
  prevRH = rhValid ? rh : prevRH;

  // body lines (stick figure overlay)
  for (let i = 0; i < POSE_EDGES.length; i++) {
    const [a, b] = POSE_EDGES[i];
    setLine(bodyLines[i], mpToWorld(pose[a]), mpToWorld(pose[b]));
  }

  // torso fill
  updateTorsoFill(pose);

  // body joints
  const jointIdxs = [0, 11, 12, 13, 14, 15, 16, 23, 24];
  for (let i = 0; i < jointIdxs.length; i++) {
    const p = mpToWorld(pose[jointIdxs[i]]);
    poseJoints[i].position.set(p[0], p[1], p[2]);
  }

  // head
  const nose = mpToWorld(pose[0]);
  headMesh.position.set(nose[0], nose[1] + 0.09, nose[2]);

  // drive avatar bones if available (arms, then hands + fingers)
  if (avatar) avatar.updateMatrixWorld(true);
  updateBoneFromJoints(POSE_TO_BONE[11], 11, 13, pose);
  updateBoneFromJoints(POSE_TO_BONE[13], 13, 15, pose);
  updateBoneFromJoints(POSE_TO_BONE[12], 12, 14, pose);
  updateBoneFromJoints(POSE_TO_BONE[14], 14, 16, pose);

  if (lhValid && lh) updateHandBonesFromLandmarks(lh, HAND_FINGER_CHAINS_LEFT);
  if (rhValid && rh) updateHandBonesFromLandmarks(rh, HAND_FINGER_CHAINS_RIGHT);

  // hands (only if valid)
  setHandVisibility("L", lhValid);
  setHandVisibility("R", rhValid);

  if (lhValid && lh) {
    for (let i = 0; i < HAND_EDGES.length; i++) {
      const [a, b] = HAND_EDGES[i];
      setLine(leftHandLines[i], mpToWorld(lh[a]), mpToWorld(lh[b]));
    }
  }

  if (rhValid && rh) {
    for (let i = 0; i < HAND_EDGES.length; i++) {
      const [a, b] = HAND_EDGES[i];
      setLine(rightHandLines[i], mpToWorld(rh[a]), mpToWorld(rh[b]));
    }
  }
}

// ---------- clip loading ----------
// Same origin when you open the viewer from serve_keypoints (port 9000) — avoids POST → HTTP 501 from plain http.server :8000
const BASE =
  typeof window !== "undefined" &&
  window.location &&
  window.location.protocol &&
  window.location.protocol.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:9000";

// ✅ IMPROVED: Clip caching to avoid repeated disk reads
const clipCache = new Map();
const loadingPromises = new Map();  // Track in-flight fetches to avoid duplicate requests

async function loadClip(gloss) {
  const url = `${BASE}/keypoints/${encodeURIComponent(gloss)}.json`;
  
  // Check cache first
  if (clipCache.has(gloss)) {
    return clipCache.get(gloss);
  }
  
  // If already loading, wait for that fetch
  if (loadingPromises.has(gloss)) {
    return loadingPromises.get(gloss);
  }
  
  // Load and cache
  const promise = (async () => {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        console.warn(`⚠️  Missing keypoints for: ${gloss}`);
        return generatePlaceholderFrame(gloss);
      }
      const data = await res.json();
      const frames = data.frames || data || [];
      clipCache.set(gloss, frames);  // Cache it
      return frames;
    } catch (e) {
      console.warn(`⚠️  Error loading clip for ${gloss}:`, e.message);
      return generatePlaceholderFrame(gloss);
    }
  })();
  
  loadingPromises.set(gloss, promise);
  
  try {
    const result = await promise;
    return result;
  } finally {
    loadingPromises.delete(gloss);
  }
}

// ✅ IMPROVED: Generate placeholder neutral pose when keypoints missing
function generatePlaceholderFrame(gloss) {
  // Return a neutral standing pose with arms at sides
  // This prevents animations from completely disappearing
  const neutralFrame = {
    pose: [
      [0.5, 0.1, 0],    // 0: nose
      [0, 0, 0], [0, 0, 0], // 1,2: eyes
      [0, 0, 0], [0, 0, 0], // 3,4: ears
      [0, 0, 0], [0, 0, 0], // 5,6: mouth
      [0, 0, 0], [0, 0, 0], // 7,8: shoulders
      [0, 0, 0],            // 9: mid-hip
      [0.4, 0.5, 0],   // 11: left shoulder
      [0.6, 0.5, 0],   // 12: right shoulder
      [0.35, 0.8, 0],  // 13: left elbow
      [0.65, 0.8, 0],  // 14: right elbow
      [0.3, 1.0, 0],   // 15: left wrist
      [0.7, 1.0, 0],   // 16: right wrist
      [0, 0, 0], [0, 0, 0], // 17,18
      [0, 0, 0], [0, 0, 0], // 19,20
      [0.4, 1.2, 0],   // 23: left hip
      [0.6, 1.2, 0],   // 24: right hip
    ],
    left_hand: null,
    right_hand: null
  };
  
  // Return 5 frames of neutral pose (0.25 seconds at 20 FPS)
  // This creates a visible pause/hold for the missing word
  return [neutralFrame, neutralFrame, neutralFrame, neutralFrame, neutralFrame];
}

// ---------- play queue ----------
let playQueue = [];
let currentFrames = [];
let frameIndex = 0;
let playing = false;
let paused = false;
let activeSequenceTokens = [];
let nextClipPromise = null;  // ✅ IMPROVED: Preload next clip

async function startSequence(tokens) {
  playQueue = [...tokens];
  activeSequenceTokens = [...tokens];
  currentFrames = [];
  frameIndex = 0;
  playing = true;
  paused = false;

  buildAvatarOnce();
  await loadNextClip();
}

async function loadNextClip() {
  // If a next clip is already being loaded, use that promise
  if (nextClipPromise) {
    try {
      currentFrames = await nextClipPromise;
      frameIndex = 0;
      nextClipPromise = null;
    } catch (e) {
      console.error("Error with preloaded clip:", e);
      currentFrames = [];
      frameIndex = 0;
    }
  }
  
  if (!playQueue.length) {
    playing = false;
    const status = document.getElementById("status");
    if (status) status.innerText = `Status: waiting for gloss...`;
    nextClipPromise = null;
    return;
  }
  
  const gloss = playQueue.shift();
  try {
    currentFrames = await loadClip(gloss);
    frameIndex = 0;
    const status = document.getElementById("status");
    
    // ✅ IMPROVED: Show which gloss is being played (including placeholders)
    if (status) {
      if (currentFrames.length <= 5) {
        status.innerText = `Status: playing ${gloss} (placeholder)`;
      } else {
        status.innerText = `Status: playing ${gloss}`;
      }
    }
    
    // ✅ IMPROVED: Preload the next clip while this one plays
    if (playQueue.length > 0) {
      const nextGloss = playQueue[0];  // Peek at next token
      nextClipPromise = loadClip(nextGloss);  // Start loading it now
    }
  } catch (e) {
    // ✅ IMPROVED: Log error but continue (don't skip tokens)
    console.error(`Error loading ${gloss}:`, e);
    currentFrames = generatePlaceholderFrame(gloss);
    frameIndex = 0;
  }
}

function tickPlayback(timeMs) {
  if (!playing || paused) return;
  if (!currentFrames.length) return;

  const now = timeMs || performance.now();
  const frameInterval = 1000 / PLAYBACK_FPS;
  if (now - lastFrameTime < frameInterval) return;
  lastFrameTime = now;

  renderFrame(currentFrames[frameIndex]);
  frameIndex++;

  if (frameIndex >= currentFrames.length) {
    loadNextClip();
  }
}

// ---------- poll state.json ----------
let lastId = 0;
let lastPollTime = 0;
const POLL_INTERVAL = 100;  // ✅ IMPROVED: Check state.json every 100ms instead of 300ms

async function pollState() {
  const now = Date.now();
  if (now - lastPollTime < POLL_INTERVAL) return;
  lastPollTime = now;
  
  try {
    const res = await fetch(`${BASE}/runtime/state.json?ts=${Date.now()}`);
    if (!res.ok) return;
    const state = await res.json();

    if (state.id && state.id !== lastId) {
      lastId = state.id;
      const tokens = (state.tokens || []).map(t => String(t).toUpperCase());
      const status = document.getElementById("status");
      if (status) status.innerText = `Status: new sentence ${tokens.join(" ")}`;
      if (tokens.length) startSequence(tokens);
    }
  } catch (_) {}
}

// manual button
const btn = document.getElementById("loadBtn");
if (btn) {
  btn.onclick = async () => {
    const val = document.getElementById("glossInput")?.value?.trim()?.toUpperCase();
    if (!val) return;
    const out = document.getElementById("glossOutput");
    if (out) out.value = val;
    startSequence(val.split(/\s+/).filter(Boolean));
  };
}

// ---------- browser mic (SpeechRecognition) ----------
const micBtn = document.getElementById("micBtn");
let recognition = null;
let micRunning = false;
let micStarting = false;  // ✅ FIX: Track if mic is currently starting

function setMicStatus(text) {
  const status = document.getElementById("status");
  if (status) status.innerText = text;
}

async function sendTextToBackend(text) {
  const res = await fetch(`${BASE}/api/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) {
    throw new Error(data.error || `Backend error (${res.status})`);
  }
  return data;
}

function setupSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.continuous = true;
  r.interimResults = true;
  r.lang = "en-US";
  return r;
}

if (micBtn) {
  recognition = setupSpeechRecognition();

  if (!recognition) {
    micBtn.disabled = true;
    micBtn.title = "SpeechRecognition not supported. Use Chrome/Edge.";
  } else {
    recognition.onstart = () => {
      micRunning = true;
      micStarting = false;  // ✅ FIX: Clear starting flag
      micBtn.innerText = "Stop Mic";
      micBtn.disabled = false;  // ✅ FIX: Re-enable button
      const listening = document.getElementById("listeningIndicator");
      if (listening) listening.innerText = "● Listening...";
      setMicStatus("Status: mic listening... speak now");
    };

    recognition.onend = () => {
      micRunning = false;
      micStarting = false;  // ✅ FIX: Clear starting flag
      micBtn.innerText = "Start Listening";
      micBtn.disabled = false;  // ✅ FIX: Re-enable button
      const listening = document.getElementById("listeningIndicator");
      if (listening) listening.innerText = "● Not listening";
      // Don't overwrite status if we are currently playing
      if (!playing) setMicStatus("Status: mic stopped");
    };

    recognition.onerror = (e) => {
      console.log("SpeechRecognition error:", e);
      micRunning = false;  // ✅ FIX: Reset on error
      micStarting = false;
      micBtn.disabled = false;  // ✅ FIX: Re-enable button
      micBtn.innerText = "Start Listening";
      setMicStatus(`Status: mic error (${e.error || "unknown"})`);
    };

    let lastFinal = "";
    recognition.onresult = async (event) => {
      let interim = "";
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0]?.transcript || "";
        if (event.results[i].isFinal) finalText += t;
        else interim += t;
      }

      if (interim.trim()) {
        const speechInput = document.getElementById("speechText");
        if (speechInput) speechInput.value = interim.trim();
        setMicStatus(`Status: hearing "${interim.trim()}" ...`);
      }

      finalText = finalText.trim();
      if (!finalText) return;
      if (finalText === lastFinal) return;
      lastFinal = finalText;

      try {
        setMicStatus(`Status: sending "${finalText}"`);
        const speechInput = document.getElementById("speechText");
        if (speechInput) speechInput.value = finalText;
        const out = await sendTextToBackend(finalText);
        const glossOut = document.getElementById("glossOutput");
        if (glossOut) glossOut.value = String(out.gloss || "").trim();
        setMicStatus(`Status: "${out.text}" → ${String(out.gloss || "").trim()}`);
      } catch (err) {
        setMicStatus(`Status: mic send failed (${err.message})`);
      }
    };

    micBtn.onclick = async () => {
      try {
        // ✅ FIX: Prevent double-clicking or rapid toggling
        if (micStarting) return;
        
        if (!micRunning) {
          micStarting = true;
          micBtn.disabled = true;
          recognition.start();
        } else {
          micBtn.disabled = true;
          recognition.stop();
        }
      } catch (e) {
        // calling start twice throws in some browsers
        console.log("Mic button error:", e);
        micStarting = false;
        micRunning = false;
        micBtn.disabled = false;
      }
    };
  }
}

const stopBtn = document.getElementById("stopBtn");
if (stopBtn) {
  stopBtn.onclick = () => {
    if (recognition && micRunning) recognition.stop();
    playing = false;
    paused = false;
    setMicStatus("Status: stopped");
  };
}

const pauseBtn = document.getElementById("pauseBtn");
if (pauseBtn) {
  pauseBtn.onclick = () => {
    paused = true;
    setMicStatus("Status: playback paused");
  };
}

const playBtn = document.getElementById("playBtn");
if (playBtn) {
  playBtn.onclick = () => {
    if (!playing && activeSequenceTokens.length) {
      startSequence(activeSequenceTokens);
    } else {
      paused = false;
      setMicStatus("Status: playback resumed");
    }
  };
}

const restartBtn = document.getElementById("restartBtn");
if (restartBtn) {
  restartBtn.onclick = () => {
    if (activeSequenceTokens.length) startSequence(activeSequenceTokens);
  };
}

setInterval(pollState, 50);  // ✅ IMPROVED: Call frequently, but pollState throttles itself to 100ms

// ---------- animate loop ----------
function animate(timeMs) {
  requestAnimationFrame(animate);
  pollState();  // ✅ IMPROVED: Also check state during render loop for faster response
  tickPlayback(timeMs);
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  const w = stageEl?.clientWidth || window.innerWidth;
  const h = stageEl?.clientHeight || window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  frameAvatarUpperBody();
});
