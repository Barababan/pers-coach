// pers.coach Shadow Training - MediaPipe Integration
// Uses MediaPipe Pose and Selfie Segmentation for overlay effect

import {
    PoseLandmarker,
    ImageSegmenter,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// Global state
const state = {
    trainerPoseLandmarker: null,
    clientPoseLandmarker: null,
    imageSegmenter: null,
    webcamRunning: false,
    trainerPose: null,
    clientPose: null,
    clientMask: null,
    trainerMask: null,
    // Separate timestamp trackers to ensure monotonically increasing values
    lastTrainerTimestamp: 0,
    lastSegTimestamp: 0,
    lastClientPoseTimestamp: 0,
    // Frame counter for throttling segmentation
    frameCount: 0,
    segmentationInterval: 3,
    // Cached aspect ratios
    cachedWebcamAspect: null,
    cachedTrainerAspect: null,
    
    // === ALIGNMENT DATA ===
    // Trainer: max height over calibration period
    trainerMaxHeight: 0,
    trainerCalibrationStart: 0,
    trainerCalibrationDone: false,
    trainerFeetY: 0.95, // Where trainer's feet are (Y coordinate)
    
    // User: smoothed values
    userHeight: 0,      // Smoothed user height
    userFeetY: 0.95,    // Smoothed user feet Y position
    userCenterX: 0.5,   // Smoothed user center X
    
    settings: {
        showShadow: true,
        showTrainerSkeleton: true,
        showClientSkeleton: true,
        autoAlign: true
    },
    fps: {
        frames: 0,
        lastTime: performance.now(),
        value: 0
    }
};

// Smoothing factor for user measurements
const SMOOTH = 0.85; // High smoothing = very stable

// Smoothing for client pose landmarks (prevent jumping)
const POSE_SMOOTH = 0.4;
let smoothedClientPose = null;
let lastValidPose = null;
let flipRejectCount = 0;

// Detect if pose is "flipped" (left/right sides swapped)
function isPoseFlipped(pose) {
    // Check shoulders: left shoulder should be to the LEFT of right shoulder (higher X in camera space)
    const ls = pose[11]; // left shoulder
    const rs = pose[12]; // right shoulder
    
    if (ls && rs && ls.visibility > 0.5 && rs.visibility > 0.5) {
        // In camera view (before mirror), left shoulder has HIGHER x than right
        // If left shoulder x < right shoulder x, the pose is flipped
        if (ls.x < rs.x) {
            return true;
        }
    }
    
    // Also check hips
    const lh = pose[23];
    const rh = pose[24];
    
    if (lh && rh && lh.visibility > 0.5 && rh.visibility > 0.5) {
        if (lh.x < rh.x) {
            return true;
        }
    }
    
    return false;
}

// Check if pose suddenly flipped compared to previous
function didPoseFlip(newPose, oldPose) {
    if (!oldPose) return false;
    
    const ls_new = newPose[11];
    const rs_new = newPose[12];
    const ls_old = oldPose[11];
    const rs_old = oldPose[12];
    
    if (!ls_new || !rs_new || !ls_old || !rs_old) return false;
    if (ls_new.visibility < 0.5 || rs_new.visibility < 0.5) return false;
    
    // Check if shoulders crossed (swapped sides)
    const oldLeftIsLeft = ls_old.x > rs_old.x;
    const newLeftIsLeft = ls_new.x > rs_new.x;
    
    if (oldLeftIsLeft !== newLeftIsLeft) {
        return true; // Flip detected!
    }
    
    // Also check if center jumped too much
    const oldCenterX = (ls_old.x + rs_old.x) / 2;
    const newCenterX = (ls_new.x + rs_new.x) / 2;
    
    if (Math.abs(newCenterX - oldCenterX) > 0.2) {
        return true; // Center jumped too much
    }
    
    return false;
}

// Smooth client pose landmarks to prevent sudden jumps
function smoothClientPose(newPose) {
    if (!newPose) return lastValidPose;
    
    // Detect flip - reject flipped frames
    if (didPoseFlip(newPose, lastValidPose)) {
        flipRejectCount++;
        // Reject up to 10 consecutive flipped frames, then accept (user actually turned)
        if (flipRejectCount < 10 && lastValidPose) {
            return lastValidPose; // Return previous valid pose
        }
    } else {
        flipRejectCount = 0;
    }
    
    if (!smoothedClientPose) {
        smoothedClientPose = newPose.map(lm => ({ ...lm }));
        lastValidPose = smoothedClientPose;
        return smoothedClientPose;
    }
    
    // Update with smoothing
    for (let i = 0; i < newPose.length; i++) {
        const newLm = newPose[i];
        const oldLm = smoothedClientPose[i];
        
        if (!newLm || !oldLm) continue;
        
        const s = POSE_SMOOTH;
        smoothedClientPose[i] = {
            ...newLm,
            x: oldLm.x * s + newLm.x * (1 - s),
            y: oldLm.y * s + newLm.y * (1 - s)
        };
    }
    
    lastValidPose = smoothedClientPose;
    return smoothedClientPose;
}

// Pre-calculated aspect ratio coefficients
let AR = { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };

function updateAspectRatioCoeffs(webcamAspect, trainerAspect) {
    if (webcamAspect > trainerAspect) {
        AR.scaleX = webcamAspect / trainerAspect;
        AR.scaleY = 1;
        AR.offsetX = (1 - AR.scaleX) / 2;
        AR.offsetY = 0;
    } else {
        AR.scaleX = 1;
        AR.scaleY = trainerAspect / webcamAspect;
        AR.offsetX = 0;
        AR.offsetY = (1 - AR.scaleY) / 2;
    }
}

// DOM Elements
const elements = {
    loadingOverlay: document.getElementById('loadingOverlay'),
    trainerVideo: document.getElementById('trainerVideo'),
    mainCanvas: document.getElementById('mainCanvas'),
    trainerCanvas: document.getElementById('trainerCanvas'),
    clientCanvas: document.getElementById('clientCanvas'),
    webcamVideo: document.getElementById('webcamVideo'),
    playBtn: document.getElementById('playBtn'),
    progressBar: document.getElementById('progressBar'),
    progressFill: document.getElementById('progressFill'),
    startBtn: document.getElementById('startBtn'),
    matchScore: document.getElementById('matchScore'),
    fpsCounter: document.getElementById('fpsCounter')
};

// Pose connections for drawing skeleton
const POSE_CONNECTIONS = [
    [11, 12], // shoulders
    [11, 13], [13, 15], // left arm
    [12, 14], [14, 16], // right arm
    [11, 23], [12, 24], // torso
    [23, 24], // hips
    [23, 25], [25, 27], [27, 29], [29, 31], // left leg
    [24, 26], [26, 28], [28, 30], [30, 32], // right leg
];

// Key landmarks for alignment
const ALIGNMENT_LANDMARKS = {
    nose: 0,
    leftShoulder: 11,
    rightShoulder: 12,
    leftHip: 23,
    rightHip: 24,
    leftAnkle: 27,
    rightAnkle: 28,
    leftHeel: 29,
    rightHeel: 30
};

// Calculate bounding box of trainer's skeleton for zoom
function getTrainerBounds(pose) {
    if (!pose) return null;
    
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    let hasVisiblePoints = false;
    
    // Check key body landmarks (not face details)
    const bodyLandmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
    
    for (const idx of bodyLandmarks) {
        if (pose[idx] && pose[idx].visibility > 0.5) {
            minX = Math.min(minX, pose[idx].x);
            maxX = Math.max(maxX, pose[idx].x);
            minY = Math.min(minY, pose[idx].y);
            maxY = Math.max(maxY, pose[idx].y);
            hasVisiblePoints = true;
        }
    }
    
    if (!hasVisiblePoints) return null;
    
    // Add padding
    const paddingX = (maxX - minX) * 0.15;
    const paddingY = (maxY - minY) * 0.1;
    
    return {
        minX: Math.max(0, minX - paddingX),
        maxX: Math.min(1, maxX + paddingX),
        minY: Math.max(0, minY - paddingY * 0.5), // Less padding on top
        maxY: Math.min(1, maxY + paddingY) // Keep feet visible
    };
}

// Get lowest visible foot point (for anchoring feet to bottom)
function getLowestFootY(pose) {
    if (!pose) return 1;
    
    const footLandmarks = [27, 28, 29, 30, 31, 32]; // ankles, heels, toes
    let maxY = 0;
    
    for (const idx of footLandmarks) {
        if (pose[idx] && pose[idx].visibility > 0.3) {
            maxY = Math.max(maxY, pose[idx].y);
        }
    }
    
    return maxY || 1;
}

// Initialize MediaPipe
async function initMediaPipe() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        // Initialize Pose Landmarker for TRAINER video
        state.trainerPoseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        // Initialize separate Pose Landmarker for CLIENT webcam
        state.clientPoseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        // Initialize Image Segmenter for body segmentation (shared between trainer and client)
        state.imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            outputCategoryMask: true,
            outputConfidenceMasks: false
        });

        console.log("MediaPipe initialized successfully");
        elements.loadingOverlay.classList.add('hidden');
        elements.startBtn.disabled = false;
        
    } catch (error) {
        console.error("Error initializing MediaPipe:", error);
        elements.loadingOverlay.querySelector('.loading-text').textContent = 
            'Ошибка загрузки. Обновите страницу.';
    }
}

// Setup webcam
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        });
        
        elements.webcamVideo.srcObject = stream;
        
        return new Promise((resolve) => {
            elements.webcamVideo.onloadedmetadata = () => {
                elements.webcamVideo.play();
                resolve();
            };
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
        alert("Не удалось получить доступ к камере. Пожалуйста, разрешите доступ.");
        throw error;
    }
}

// Process trainer video frame
function processTrainerFrame(timestamp) {
    if (!state.trainerPoseLandmarker || elements.trainerVideo.paused) return;
    if (elements.trainerVideo.readyState < 2) return; // Not enough data
    
    state.frameCount++;
    
    // Ensure strictly increasing timestamp
    const ts = Math.max(timestamp, state.lastTrainerTimestamp + 1);
    state.lastTrainerTimestamp = ts;
    
    try {
        const results = state.trainerPoseLandmarker.detectForVideo(elements.trainerVideo, ts);
        
        if (results.landmarks && results.landmarks.length > 0) {
            state.trainerPose = results.landmarks[0];
        } else {
            state.trainerPose = null;
        }
    } catch (e) {
        console.warn("Trainer pose error:", e.message);
    }
}

// Process segmentation for both trainer and client (alternating)
function processSegmentation(timestamp) {
    if (!state.imageSegmenter) return;
    
    state.frameCount++;
    
    // Only run every N frames for performance
    if (state.frameCount % state.segmentationInterval !== 0) return;
    
    const segTs = Math.max(timestamp, state.lastSegTimestamp + 1);
    state.lastSegTimestamp = segTs;
    
    // Determine what to segment
    const trainerReady = elements.trainerVideo.readyState >= 2 && !elements.trainerVideo.paused;
    const clientReady = state.webcamRunning && elements.webcamVideo.readyState >= 2;
    
    // Alternate between trainer and client when both are ready
    // Otherwise just segment what's available
    const cycle = Math.floor(state.frameCount / state.segmentationInterval) % 2;
    
    try {
        if (trainerReady && clientReady) {
            // Both ready - alternate
            if (cycle === 0) {
                const segResults = state.imageSegmenter.segmentForVideo(elements.trainerVideo, segTs);
                if (segResults.categoryMask) {
                    state.trainerMask = segResults.categoryMask;
                }
            } else {
                const segResults = state.imageSegmenter.segmentForVideo(elements.webcamVideo, segTs);
                if (segResults.categoryMask) {
                    state.clientMask = segResults.categoryMask;
                }
            }
        } else if (trainerReady) {
            // Only trainer ready
            const segResults = state.imageSegmenter.segmentForVideo(elements.trainerVideo, segTs);
            if (segResults.categoryMask) {
                state.trainerMask = segResults.categoryMask;
            }
        } else if (clientReady) {
            // Only client ready
            const segResults = state.imageSegmenter.segmentForVideo(elements.webcamVideo, segTs);
            if (segResults.categoryMask) {
                state.clientMask = segResults.categoryMask;
            }
        }
    } catch (e) {
        console.warn("Segmentation error:", e.message);
    }
}

// Process webcam frame for pose only
function processWebcamFrame(timestamp) {
    if (!state.clientPoseLandmarker || !state.webcamRunning) return;
    if (elements.webcamVideo.readyState < 2) return; // Not enough data
    
    // Ensure strictly increasing timestamps
    const poseTs = Math.max(timestamp, state.lastClientPoseTimestamp + 1);
    state.lastClientPoseTimestamp = poseTs;
    
    try {
        // Get pose (every frame for responsive skeleton)
        const poseResults = state.clientPoseLandmarker.detectForVideo(elements.webcamVideo, poseTs);
        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
            state.clientPose = poseResults.landmarks[0];
        } else {
            state.clientPose = null;
        }
    } catch (e) {
        console.warn("Client pose error:", e.message);
    }
}

// ============================================
// STEP 1: Measure trainer height (calibrate over 2 seconds)
// ============================================
function updateTrainerMeasurements(pose) {
    if (!pose) return;
    
    const now = performance.now();
    
    // Start calibration timer
    if (state.trainerCalibrationStart === 0) {
        state.trainerCalibrationStart = now;
    }
    
    // Get current height measurement
    const topY = getTopY(pose);
    const bottomY = getBottomY(pose);
    const height = bottomY - topY;
    
    if (height > 0.1) {
        // During calibration (first 2 seconds): track maximum height
        if (now - state.trainerCalibrationStart < 2000) {
            if (height > state.trainerMaxHeight) {
                state.trainerMaxHeight = height;
                state.trainerFeetY = bottomY;
            }
        } else {
            state.trainerCalibrationDone = true;
        }
    }
}

// ============================================
// STEP 2: Measure user height and feet (smoothed)
// ============================================
function updateUserMeasurements(pose) {
    if (!pose) return;
    
    // Get current measurements
    const topY = getTopY(pose);
    const bottomY = getBottomY(pose);
    const height = bottomY - topY;
    const centerX = getCenterX(pose);
    
    if (height > 0.1 && centerX !== null) {
        // Smooth updates (high smoothing = stable)
        if (state.userHeight === 0) {
            // First measurement
            state.userHeight = height;
            state.userFeetY = bottomY;
            state.userCenterX = centerX;
        } else {
            state.userHeight = state.userHeight * SMOOTH + height * (1 - SMOOTH);
            state.userFeetY = state.userFeetY * SMOOTH + bottomY * (1 - SMOOTH);
            state.userCenterX = state.userCenterX * SMOOTH + centerX * (1 - SMOOTH);
        }
    }
}

// Get top Y (head/nose position)
function getTopY(pose) {
    const nose = pose[0];
    if (nose && nose.visibility > 0.5) {
        return nose.y;
    }
    
    // Fallback: estimate from shoulders
    const ls = pose[11], rs = pose[12];
    if (ls && rs && ls.visibility > 0.3 && rs.visibility > 0.3) {
        const shoulderY = (ls.y + rs.y) / 2;
        // Head is ~25% of torso height above shoulders
        const lh = pose[23], rh = pose[24];
        if (lh && rh && lh.visibility > 0.3) {
            const hipY = (lh.y + rh.y) / 2;
            const torsoH = hipY - shoulderY;
            return shoulderY - torsoH * 0.25;
        }
        return shoulderY - 0.05; // Small offset
    }
    
    return 0.1; // Default to near top
}

// Get bottom Y (feet position)
function getBottomY(pose) {
    // Check all foot landmarks
    const footIndices = [27, 28, 29, 30, 31, 32];
    let maxY = 0;
    
    for (const i of footIndices) {
        const lm = pose[i];
        if (lm && lm.visibility > 0.3 && lm.y > maxY) {
            maxY = lm.y;
        }
    }
    
    if (maxY > 0.3) return maxY;
    
    // Estimate from hips
    const lh = pose[23], rh = pose[24];
    if (lh && rh && lh.visibility > 0.3 && rh.visibility > 0.3) {
        const hipY = (lh.y + rh.y) / 2;
        const ls = pose[11], rs = pose[12];
        if (ls && rs && ls.visibility > 0.3) {
            const shoulderY = (ls.y + rs.y) / 2;
            const torsoH = hipY - shoulderY;
            return hipY + torsoH * 1.1; // Legs ~ torso height
        }
        return hipY + 0.3;
    }
    
    return 0.95; // Default to near bottom
}

// Get center X (hip center, stable)
function getCenterX(pose) {
    const lh = pose[23], rh = pose[24];
    if (lh && rh && lh.visibility > 0.3 && rh.visibility > 0.3) {
        return (lh.x + rh.x) / 2;
    }
    const ls = pose[11], rs = pose[12];
    if (ls && rs && ls.visibility > 0.3 && rs.visibility > 0.3) {
        return (ls.x + rs.x) / 2;
    }
    return null;
}

// ============================================
// STEP 3: Calculate simple scale and offset
// ============================================
function getAlignmentParams() {
    if (!state.trainerCalibrationDone || state.userHeight < 0.1 || state.trainerMaxHeight < 0.1) {
        return { scale: 1, offsetX: 0, offsetY: 0 };
    }
    
    // Scale: trainer height / user height
    const scale = state.trainerMaxHeight / state.userHeight;
    const clampedScale = Math.max(0.5, Math.min(2.0, scale));
    
    // Offset: move user's feet to trainer's feet
    // User feet Y after scaling: userFeetY doesn't change with scale since we scale around feet
    // But we need to position correctly...
    
    // Simple approach: user's feet should be at trainer's feet position
    const offsetY = state.trainerFeetY - state.userFeetY;
    
    // X offset: center user on trainer (use 0.5 as default trainer center)
    const trainerCenterX = 0.5;
    const userCenterXMirrored = 1 - state.userCenterX; // Mirror
    const offsetX = trainerCenterX - userCenterXMirrored;
    
    return { scale: clampedScale, offsetX, offsetY };
}

// Get center point of pose (average of shoulders and hips)
function getPoseCenter(pose) {
    const ls = pose[ALIGNMENT_LANDMARKS.leftShoulder];
    const rs = pose[ALIGNMENT_LANDMARKS.rightShoulder];
    const lh = pose[ALIGNMENT_LANDMARKS.leftHip];
    const rh = pose[ALIGNMENT_LANDMARKS.rightHip];
    
    // Check if all landmarks are visible enough
    if (!ls || !rs || !lh || !rh) return null;
    if (ls.visibility < 0.3 || rs.visibility < 0.3 || lh.visibility < 0.3 || rh.visibility < 0.3) {
        return null;
    }
    
    return {
        x: (ls.x + rs.x + lh.x + rh.x) / 4,
        y: (ls.y + rs.y + lh.y + rh.y) / 4
    };
}

// Get size of pose (distance from shoulders to hips)
function getPoseSize(pose) {
    const ls = pose[ALIGNMENT_LANDMARKS.leftShoulder];
    const rs = pose[ALIGNMENT_LANDMARKS.rightShoulder];
    const lh = pose[ALIGNMENT_LANDMARKS.leftHip];
    const rh = pose[ALIGNMENT_LANDMARKS.rightHip];
    
    if (!ls || !rs || !lh || !rh) return 0;
    
    const shoulderWidth = Math.sqrt(Math.pow(rs.x - ls.x, 2) + Math.pow(rs.y - ls.y, 2));
    const torsoHeight = Math.sqrt(
        Math.pow((rs.x + ls.x) / 2 - (rh.x + lh.x) / 2, 2) +
        Math.pow((rs.y + ls.y) / 2 - (rh.y + lh.y) / 2, 2)
    );
    
    return shoulderWidth + torsoHeight;
}

// Calculate match score between poses (with transform applied to client)
// Transform: clientX_transformed = ax * clientX + bx
function calculateMatchScoreFast(trainerPose, clientPose, ax, bx, ay, by) {
    if (!trainerPose || !clientPose) return 0;
    
    let totalDistance = 0;
    let count = 0;
    
    // Key body landmarks (skip face, hands, feet details)
    const keyLandmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    
    for (let i = 0; i < keyLandmarks.length; i++) {
        const idx = keyLandmarks[i];
        const t = trainerPose[idx];
        const c = clientPose[idx];
        if (t && c && t.visibility > 0.5 && c.visibility > 0.5) {
            // Apply transform to client pose inline
            const cx = ax * c.x + bx;
            const cy = ay * c.y + by;
            const dx = t.x - cx;
            const dy = t.y - cy;
            totalDistance += Math.sqrt(dx * dx + dy * dy);
            count++;
        }
    }
    
    if (count === 0) return 0;
    
    const avgDistance = totalDistance / count;
    return Math.max(0, Math.min(100, Math.round((1 - avgDistance * 2) * 100)));
}

// Draw skeleton on canvas (for trainer - no transform)
function drawSkeleton(ctx, pose, color) {
    if (!pose) return;
    
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    
    // Draw connections
    for (const [start, end] of POSE_CONNECTIONS) {
        const p1 = pose[start], p2 = pose[end];
        if (p1 && p2 && p1.visibility > 0.5 && p2.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(p1.x * w, p1.y * h);
            ctx.lineTo(p2.x * w, p2.y * h);
            ctx.stroke();
        }
    }
    
    // Draw landmarks
    for (let i = 0; i < pose.length; i++) {
        const p = pose[i];
        if (p && p.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(p.x * w, p.y * h, 5, 0, 6.283);
            ctx.fill();
        }
    }
}

// Fast skeleton drawing with pre-calculated linear transform
// Transform: screenX = ax * poseX + bx, screenY = ay * poseY + by
function drawSkeletonFast(ctx, pose, color, ax, bx, ay, by, w, h) {
    if (!pose) return;
    
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    
    // Pre-calculate transformed coordinates for all visible landmarks
    const coords = new Array(33);
    for (let i = 0; i < pose.length; i++) {
        const p = pose[i];
        if (p && p.visibility > 0.5) {
            coords[i] = {
                x: (ax * p.x + bx) * w,
                y: (ay * p.y + by) * h
            };
        }
    }
    
    // Draw connections using pre-calculated coords
    for (const [start, end] of POSE_CONNECTIONS) {
        const c1 = coords[start], c2 = coords[end];
        if (c1 && c2) {
            ctx.beginPath();
            ctx.moveTo(c1.x, c1.y);
            ctx.lineTo(c2.x, c2.y);
            ctx.stroke();
        }
    }
    
    // Draw landmarks
    for (let i = 0; i < coords.length; i++) {
        const c = coords[i];
        if (c) {
            ctx.beginPath();
            ctx.arc(c.x, c.y, 5, 0, 6.283);
            ctx.fill();
        }
    }
}

// Cached canvases for cutout rendering
let cutoutCanvas = null;
let cutoutCtx = null;
let webcamCanvas = null;
let webcamCtx = null;

// Draw user cutout (webcam image masked by segmentation)
function drawShadow(ctx, mask) {
    if (!mask || !state.settings.showShadow) return;
    
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    const webcamVideo = elements.webcamVideo;
    
    if (!webcamVideo || webcamVideo.readyState < 2) return;
    
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    
    // Initialize canvases if needed
    if (!cutoutCanvas || cutoutCanvas.width !== maskWidth || cutoutCanvas.height !== maskHeight) {
        cutoutCanvas = document.createElement('canvas');
        cutoutCanvas.width = maskWidth;
        cutoutCanvas.height = maskHeight;
        cutoutCtx = cutoutCanvas.getContext('2d', { willReadFrequently: true });
        
        webcamCanvas = document.createElement('canvas');
        webcamCanvas.width = maskWidth;
        webcamCanvas.height = maskHeight;
        webcamCtx = webcamCanvas.getContext('2d', { willReadFrequently: true });
    }
    
    // Draw webcam frame to temp canvas
    webcamCtx.drawImage(webcamVideo, 0, 0, maskWidth, maskHeight);
    const webcamImageData = webcamCtx.getImageData(0, 0, maskWidth, maskHeight);
    const webcamPixels = webcamImageData.data;
    
    // Get mask data
    const maskData = mask.getAsUint8Array();
    
    // Create cutout: webcam pixels where mask is person, transparent elsewhere
    const cutoutImageData = cutoutCtx.createImageData(maskWidth, maskHeight);
    const cutoutPixels = cutoutImageData.data;
    
    for (let i = 0, len = maskData.length; i < len; i++) {
        const idx = i << 2;
        if (maskData[i] === 0) { // Person
            cutoutPixels[idx] = webcamPixels[idx];         // R
            cutoutPixels[idx + 1] = webcamPixels[idx + 1]; // G
            cutoutPixels[idx + 2] = webcamPixels[idx + 2]; // B
            cutoutPixels[idx + 3] = 220; // Slightly transparent
        }
        // Background stays transparent (0)
    }
    
    cutoutCtx.putImageData(cutoutImageData, 0, 0);
    
    // Calculate drawing dimensions using global AR coefficients
    const drawWidth = canvasWidth * AR.scaleX;
    const drawHeight = canvasHeight * AR.scaleY;
    const drawX = AR.offsetX * canvasWidth;
    const drawY = AR.offsetY * canvasHeight;

    // Draw to main canvas (mirrored)
    ctx.save();
    ctx.translate(canvasWidth, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(cutoutCanvas, drawX, drawY, drawWidth, drawHeight);
    ctx.restore();
}

// Cached canvases for trainer cutout rendering
let trainerCutoutCanvas = null;
let trainerCutoutCtx = null;
let trainerVideoCanvas = null;
let trainerVideoCtx = null;

// Draw trainer cutout (trainer video masked by segmentation)
function drawTrainerCutout(ctx, mask, video) {
    if (!mask) return;
    
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    
    if (!video || video.readyState < 2) return;
    
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    
    // Initialize canvases if needed
    if (!trainerCutoutCanvas || trainerCutoutCanvas.width !== maskWidth || trainerCutoutCanvas.height !== maskHeight) {
        trainerCutoutCanvas = document.createElement('canvas');
        trainerCutoutCanvas.width = maskWidth;
        trainerCutoutCanvas.height = maskHeight;
        trainerCutoutCtx = trainerCutoutCanvas.getContext('2d', { willReadFrequently: true });
        
        trainerVideoCanvas = document.createElement('canvas');
        trainerVideoCanvas.width = maskWidth;
        trainerVideoCanvas.height = maskHeight;
        trainerVideoCtx = trainerVideoCanvas.getContext('2d', { willReadFrequently: true });
    }
    
    // Draw video frame to temp canvas
    trainerVideoCtx.drawImage(video, 0, 0, maskWidth, maskHeight);
    const videoImageData = trainerVideoCtx.getImageData(0, 0, maskWidth, maskHeight);
    const videoPixels = videoImageData.data;
    
    // Get mask data
    const maskData = mask.getAsUint8Array();
    
    // Create cutout: video pixels where mask is person, transparent elsewhere
    const cutoutImageData = trainerCutoutCtx.createImageData(maskWidth, maskHeight);
    const cutoutPixels = cutoutImageData.data;
    
    for (let i = 0, len = maskData.length; i < len; i++) {
        const idx = i << 2;
        if (maskData[i] === 0) { // Person
            cutoutPixels[idx] = videoPixels[idx];         // R
            cutoutPixels[idx + 1] = videoPixels[idx + 1]; // G
            cutoutPixels[idx + 2] = videoPixels[idx + 2]; // B
            cutoutPixels[idx + 3] = 255; // Full opacity
        }
        // Background stays transparent (0)
    }
    
    trainerCutoutCtx.putImageData(cutoutImageData, 0, 0);
    
    // Draw to canvas (no mirroring for trainer, fit to canvas)
    ctx.drawImage(trainerCutoutCanvas, 0, 0, canvasWidth, canvasHeight);
}

// Draw client cutout to separate canvas (with mirroring)
function drawClientCutout(ctx, mask, video) {
    if (!mask) return;
    
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    
    if (!video || video.readyState < 2) return;
    
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    
    // Reuse cutout canvases from drawShadow
    if (!cutoutCanvas || cutoutCanvas.width !== maskWidth || cutoutCanvas.height !== maskHeight) {
        cutoutCanvas = document.createElement('canvas');
        cutoutCanvas.width = maskWidth;
        cutoutCanvas.height = maskHeight;
        cutoutCtx = cutoutCanvas.getContext('2d', { willReadFrequently: true });
        
        webcamCanvas = document.createElement('canvas');
        webcamCanvas.width = maskWidth;
        webcamCanvas.height = maskHeight;
        webcamCtx = webcamCanvas.getContext('2d', { willReadFrequently: true });
    }
    
    // Draw webcam frame to temp canvas
    webcamCtx.drawImage(video, 0, 0, maskWidth, maskHeight);
    const webcamImageData = webcamCtx.getImageData(0, 0, maskWidth, maskHeight);
    const webcamPixels = webcamImageData.data;
    
    // Get mask data
    const maskData = mask.getAsUint8Array();
    
    // Create cutout: webcam pixels where mask is person, transparent elsewhere
    const cutoutImageData = cutoutCtx.createImageData(maskWidth, maskHeight);
    const cutoutPixels = cutoutImageData.data;
    
    for (let i = 0, len = maskData.length; i < len; i++) {
        const idx = i << 2;
        if (maskData[i] === 0) { // Person
            cutoutPixels[idx] = webcamPixels[idx];         // R
            cutoutPixels[idx + 1] = webcamPixels[idx + 1]; // G
            cutoutPixels[idx + 2] = webcamPixels[idx + 2]; // B
            cutoutPixels[idx + 3] = 255; // Full opacity
        }
        // Background stays transparent (0)
    }
    
    cutoutCtx.putImageData(cutoutImageData, 0, 0);
    
    // Draw to canvas (mirrored for selfie view)
    ctx.save();
    ctx.translate(canvasWidth, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(cutoutCanvas, 0, 0, canvasWidth, canvasHeight);
    ctx.restore();
}

// Main render loop
function render() {
    const timestamp = performance.now();
    
    // Update FPS
    state.fps.frames++;
    if (timestamp - state.fps.lastTime >= 1000) {
        state.fps.value = state.fps.frames;
        state.fps.frames = 0;
        state.fps.lastTime = timestamp;
        elements.fpsCounter.textContent = state.fps.value;
    }
    
    // Process frames
    processTrainerFrame(timestamp);
    if (state.webcamRunning) {
        processWebcamFrame(timestamp);
    }
    
    // Process segmentation (alternates between trainer and client)
    if (state.settings.showShadow) {
        processSegmentation(timestamp);
    }
    
    // Get video dimensions
    const videoWidth = elements.trainerVideo.videoWidth || 1080;
    const videoHeight = elements.trainerVideo.videoHeight || 1920;
    const videoAspect = videoWidth / videoHeight;
    
    // Get webcam dimensions
    const webcamWidth = elements.webcamVideo.videoWidth || 640;
    const webcamHeight = elements.webcamVideo.videoHeight || 480;
    const webcamAspect = webcamWidth / webcamHeight;
    
    // Cache aspect ratios
    if (webcamWidth > 100 && webcamHeight > 100) {
        state.cachedWebcamAspect = webcamAspect;
    }
    if (videoWidth > 100 && videoHeight > 100) {
        state.cachedTrainerAspect = videoAspect;
    }
    
    // === TRAINER CANVAS ===
    const trainerCtx = elements.trainerCanvas.getContext('2d');
    const trainerPanel = elements.trainerCanvas.parentElement;
    const trainerPanelW = trainerPanel.clientWidth;
    const trainerPanelH = trainerPanel.clientHeight;
    
    // Calculate canvas size to fit panel while maintaining video aspect ratio
    let trainerCanvasW, trainerCanvasH;
    if (videoAspect > trainerPanelW / trainerPanelH) {
        trainerCanvasW = trainerPanelW;
        trainerCanvasH = trainerPanelW / videoAspect;
    } else {
        trainerCanvasH = trainerPanelH;
        trainerCanvasW = trainerPanelH * videoAspect;
    }
    
    // Set trainer canvas size
    if (elements.trainerCanvas.width !== Math.round(trainerCanvasW) || 
        elements.trainerCanvas.height !== Math.round(trainerCanvasH)) {
        elements.trainerCanvas.width = Math.round(trainerCanvasW);
        elements.trainerCanvas.height = Math.round(trainerCanvasH);
    }
    
    // Clear trainer canvas with gray background
    trainerCtx.fillStyle = '#3a3a3a';
    trainerCtx.fillRect(0, 0, elements.trainerCanvas.width, elements.trainerCanvas.height);
    
    // Draw trainer cutout (video masked by segmentation)
    if (state.trainerMask && state.settings.showShadow) {
        drawTrainerCutout(trainerCtx, state.trainerMask, elements.trainerVideo);
    } else if (elements.trainerVideo.readyState >= 2) {
        // Fallback: draw full video if no mask yet
        trainerCtx.drawImage(elements.trainerVideo, 0, 0, elements.trainerCanvas.width, elements.trainerCanvas.height);
    }
    
    // Draw trainer skeleton
    if (state.settings.showTrainerSkeleton && state.trainerPose) {
        drawSkeleton(trainerCtx, state.trainerPose, 'rgba(255, 100, 100, 0.9)');
    }
    
    // === CLIENT CANVAS ===
    const clientCtx = elements.clientCanvas.getContext('2d');
    const clientPanel = elements.clientCanvas.parentElement;
    const clientPanelW = clientPanel.clientWidth;
    const clientPanelH = clientPanel.clientHeight;
    
    // For client, we want to match trainer's aspect ratio for better comparison
    // or use webcam aspect if no trainer
    const targetAspect = videoAspect;
    
    let clientCanvasW, clientCanvasH;
    if (targetAspect > clientPanelW / clientPanelH) {
        clientCanvasW = clientPanelW;
        clientCanvasH = clientPanelW / targetAspect;
    } else {
        clientCanvasH = clientPanelH;
        clientCanvasW = clientPanelH * targetAspect;
    }
    
    // Set client canvas size
    if (elements.clientCanvas.width !== Math.round(clientCanvasW) || 
        elements.clientCanvas.height !== Math.round(clientCanvasH)) {
        elements.clientCanvas.width = Math.round(clientCanvasW);
        elements.clientCanvas.height = Math.round(clientCanvasH);
    }
    
    // Clear client canvas with gray background
    clientCtx.fillStyle = '#3a3a3a';
    clientCtx.fillRect(0, 0, elements.clientCanvas.width, elements.clientCanvas.height);
    
    // Calculate aspect ratio coefficients for client
    updateAspectRatioCoeffs(state.cachedWebcamAspect || webcamAspect, targetAspect);
    
    // Draw client cutout (webcam masked by segmentation)
    if (state.clientMask && state.settings.showShadow && state.webcamRunning) {
        drawClientCutout(clientCtx, state.clientMask, elements.webcamVideo);
    } else if (state.webcamRunning && elements.webcamVideo.readyState >= 2) {
        // Fallback: draw mirrored webcam if no mask yet
        clientCtx.save();
        clientCtx.translate(elements.clientCanvas.width, 0);
        clientCtx.scale(-1, 1);
        
        // Crop webcam to match target aspect
        const srcW = elements.webcamVideo.videoWidth;
        const srcH = elements.webcamVideo.videoHeight;
        const srcAspect = srcW / srcH;
        
        let sx = 0, sy = 0, sw = srcW, sh = srcH;
        if (srcAspect > targetAspect) {
            sw = srcH * targetAspect;
            sx = (srcW - sw) / 2;
        } else {
            sh = srcW / targetAspect;
            sy = (srcH - sh) / 2;
        }
        
        clientCtx.drawImage(elements.webcamVideo, sx, sy, sw, sh, 
                           0, 0, elements.clientCanvas.width, elements.clientCanvas.height);
        clientCtx.restore();
    }
    
    // Draw client skeleton (mirrored)
    if (state.settings.showClientSkeleton && state.clientPose && state.webcamRunning) {
        const smoothedPose = smoothClientPose(state.clientPose);
        
        // Transform: mirror + aspect ratio correction
        const ax = -AR.scaleX;
        const bx = AR.offsetX + AR.scaleX;
        const ay = AR.scaleY;
        const by = AR.offsetY;
        drawSkeletonFast(clientCtx, smoothedPose, 'rgba(0, 255, 136, 0.9)', ax, bx, ay, by, 
                        elements.clientCanvas.width, elements.clientCanvas.height);
    }
    
    // Update match score (with aspect ratio correction)
    if (state.trainerPose && state.clientPose) {
        const ax = -AR.scaleX;
        const bx = AR.offsetX + AR.scaleX;
        const ay = AR.scaleY;
        const by = AR.offsetY;
        const score = calculateMatchScoreFast(state.trainerPose, state.clientPose, ax, bx, ay, by);
        elements.matchScore.textContent = score + '%';
        elements.matchScore.style.color = score > 70 ? '#00ff88' : score > 40 ? '#ffaa00' : '#ff4444';
    }
    
    // Update progress bar
    if (elements.trainerVideo.duration) {
        const progress = (elements.trainerVideo.currentTime / elements.trainerVideo.duration) * 100;
        elements.progressFill.style.width = progress + '%';
    }
    
    requestAnimationFrame(render);
}

// Event handlers
function setupEventHandlers() {
    // Play/Pause button
    elements.playBtn.addEventListener('click', () => {
        if (elements.trainerVideo.paused) {
            elements.trainerVideo.play();
            elements.playBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>';
        } else {
            elements.trainerVideo.pause();
            elements.playBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
        }
    });
    
    // Progress bar click
    elements.progressBar.addEventListener('click', (e) => {
        const rect = elements.progressBar.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        elements.trainerVideo.currentTime = pos * elements.trainerVideo.duration;
    });
    
    // Start button
    elements.startBtn.addEventListener('click', async () => {
        if (!state.webcamRunning) {
            await setupWebcam();
            state.webcamRunning = true;
            elements.startBtn.textContent = 'Остановить';
            elements.trainerVideo.play();
            elements.playBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>';
        } else {
            state.webcamRunning = false;
            // Reset state
            state.cachedWebcamAspect = null;
            state.trainerMaxHeight = 0;
            state.trainerCalibrationStart = 0;
            state.trainerCalibrationDone = false;
            state.userHeight = 0;
            state.userFeetY = 0.95;
            state.userCenterX = 0.5;
            state.clientMask = null;
            smoothedClientPose = null; // Reset pose smoothing
            lastValidPose = null;
            flipRejectCount = 0;
            const stream = elements.webcamVideo.srcObject;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            elements.webcamVideo.srcObject = null;
            elements.startBtn.textContent = 'Начать тренировку';
            elements.trainerVideo.pause();
            elements.playBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
        }
    });
    
    // Toggle switches
    document.querySelectorAll('.toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            const setting = toggle.dataset.setting;
            toggle.classList.toggle('active');
            state.settings[setting] = toggle.classList.contains('active');
        });
    });
}

// Initialize app
async function init() {
    setupEventHandlers();
    await initMediaPipe();
    render();
}

init();

