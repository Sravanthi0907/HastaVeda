// webcam.js - Real-time Webcam Mudra Recognition Engine
document.addEventListener("DOMContentLoaded", () => {
    // UI elements
    const video = document.getElementById("webcamVideo");
    const canvas = document.getElementById("skeletonCanvas");
    const ctx = canvas.getContext("2d");
    
    const btnStart = document.getElementById("btnStartWebcam");
    const btnStop = document.getElementById("btnStopWebcam");
    const spinner = document.getElementById("loadingSpinner");
    
    const liveMudraName = document.getElementById("liveMudraName");
    const liveMudraStatus = document.getElementById("liveMudraStatus");
    const liveMudraConf = document.getElementById("liveMudraConf");
    const liveMudraBar = document.getElementById("liveMudraBar");
    
    // Prediction state variables
    let cameraActive = false;
    let cameraStream = null;
    let mediaPipeCamera = null;
    
    // Performance rate limiter: call backend every 150ms (instead of 30 times a second)
    let lastPredictTime = 0;
    const PREDICT_INTERVAL = 150; 
    
    // Drawing colors
    const COLOR_GOLD = "#D4AF37";
    const COLOR_MAROON = "#6A0D25";
    
    // Initialize MediaPipe Solutions
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.75,
        minTrackingConfidence: 0.75
    });

    // Hook results handlers
    hands.onResults(onHandResults);

    // Start Webcam
    btnStart.addEventListener("click", async () => {
        spinner.classList.remove("d-none");
        btnStart.disabled = true;
        
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, frameRate: { ideal: 30 } },
                audio: false
            });
            video.srcObject = cameraStream;
            cameraActive = true;
            
            // Adjust canvas size to match video
            video.addEventListener("loadedmetadata", () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                spinner.classList.add("d-none");
                btnStop.disabled = false;
            });
            
            // Setup MediaPipe Camera Utility
            mediaPipeCamera = new Camera(video, {
                onFrame: async () => {
                    if (cameraActive) {
                        // Forward camera frames only to MediaPipe Hands pipeline
                        await hands.send({ image: video });
                    }
                },
                width: 640,
                height: 480
            });
            mediaPipeCamera.start();
            
            liveMudraStatus.className = "badge bg-warning text-dark px-3 py-2 rounded-pill fs-6";
            liveMudraStatus.innerText = "Analyzing Feed...";
            
        } catch (err) {
            console.error("Webcam initiation failed:", err);
            alert("Unable to open your camera. Please ensure permissions are granted and no other application is using it.");
            spinner.classList.add("d-none");
            btnStart.disabled = false;
        }
    });

    // Stop Webcam
    btnStop.addEventListener("click", () => {
        cameraActive = false;
        btnStop.disabled = true;
        
        if (mediaPipeCamera) {
            mediaPipeCamera.stop();
        }
        
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }
        
        video.srcObject = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        btnStart.disabled = false;
        spinner.classList.add("d-none");
        
        // Reset dashboard metrics
        liveMudraName.innerText = "None";
        liveMudraStatus.className = "badge bg-secondary px-3 py-2 rounded-pill fs-6";
        liveMudraStatus.innerText = "Waiting for Camera";
        liveMudraConf.innerText = "0%";
        liveMudraBar.style.width = "0%";
    });

    // Drawing helpers
    function drawLine(p1, p2, color, thickness) {
        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.stroke();
    }

    function drawHandSkeleton(landmarks, color) {
        // Draw MediaPipe hand skeleton connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12],
            [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [17, 18], [18, 19], [19, 20],
            [0, 17]
        ];
        
        // Draw connection lines
        connections.forEach(([p1, p2]) => {
            drawLine(landmarks[p1], landmarks[p2], color, 2.5);
        });

        // Draw joint circles
        landmarks.forEach(lm => {
            ctx.beginPath();
            ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
            ctx.fillStyle = COLOR_GOLD;
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = COLOR_MAROON;
            ctx.stroke();
        });
    }

    // Process Hand landmarks
    async function onHandResults(results) {
        // Clear canvas overlay first on each frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            liveMudraName.innerText = "No Hand Detected";
            if (liveMudraStatus) {
                liveMudraStatus.className = "badge bg-secondary px-3 py-2 rounded-pill fs-6";
                liveMudraStatus.innerText = "Waiting...";
            }
            liveMudraConf.innerText = "0%";
            liveMudraBar.style.width = "0%";
            return;
        }

        // Draw Hand landmarks and skeletons on canvas overlay
        results.multiHandLandmarks.forEach(handLandmarks => {
            // Draw classical red-maroon skeleton
            drawHandSkeleton(handLandmarks, COLOR_MAROON);
            
            // Calculate Bounding Box
            const xs = handLandmarks.map(lm => lm.x * canvas.width);
            const ys = handLandmarks.map(lm => lm.y * canvas.height);
            const xMin = Math.max(0, Math.min(...xs) - 15);
            const xMax = Math.min(canvas.width, Math.max(...xs) + 15);
            const yMin = Math.max(0, Math.min(...ys) - 15);
            const yMax = Math.min(canvas.height, Math.max(...ys) + 15);
            
            // Draw golden frame bounding box
            ctx.strokeStyle = COLOR_GOLD;
            ctx.lineWidth = 2;
            ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
        });

        // Rate-limit predictions sent to the backend Flask model
        const now = Date.now();
        if (now - lastPredictTime >= PREDICT_INTERVAL) {
            lastPredictTime = now;
            
            // For prediction, extract landmarks scaled to reference 640x480 frame matching dataset scale
            const hand = results.multiHandLandmarks[0];
            const xCoords = hand.map(lm => lm.x * 640);
            const yCoords = hand.map(lm => lm.y * 480);
            const landmarks42 = xCoords.concat(yCoords);
            
            try {
                const response = await fetch("/api/predict-landmarks", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ landmarks: landmarks42 })
                });
                const data = await response.json();
                
                if (data.mudra) {
                    // Update dashboard metrics
                    const isRecognized = data.status === "Recognized";
                    liveMudraName.innerText = data.mudra.charAt(0).toUpperCase() + data.mudra.slice(1);
                    liveMudraStatus.className = isRecognized ? "badge bg-success px-3 py-2 rounded-pill fs-6" : "badge bg-danger px-3 py-2 rounded-pill fs-6";
                    liveMudraStatus.innerText = data.status;
                    
                    const confVal = Math.round(data.confidence * 100);
                    liveMudraConf.innerText = `${confVal}%`;
                    liveMudraBar.style.width = `${confVal}%`;
                }
            } catch (err) {
                console.error("Mudra predict API call failed:", err);
            }
        }
    }
});
