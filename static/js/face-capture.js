/**
 * face-capture.js
 * Shared webcam helper used on both the signup and login pages.
 * Starts the camera, captures a still frame as a base64 JPEG, and stops it.
 */

async function startCamera(videoElement) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 480 }, height: { ideal: 480 } },
    audio: false,
  });
  videoElement.srcObject = stream;
  await videoElement.play();
  return stream;
}

function captureFrame(videoElement, canvasElement) {
  const size = Math.min(videoElement.videoWidth, videoElement.videoHeight);
  canvasElement.width = size;
  canvasElement.height = size;
  const ctx = canvasElement.getContext("2d");

  // Crop to a centered square so the saved image matches the circular preview
  const offsetX = (videoElement.videoWidth - size) / 2;
  const offsetY = (videoElement.videoHeight - size) / 2;
  ctx.drawImage(videoElement, offsetX, offsetY, size, size, 0, 0, size, size);

  return canvasElement.toDataURL("image/jpeg", 0.9);
}

function stopCamera(stream) {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
}
