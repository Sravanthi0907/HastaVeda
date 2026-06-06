// upload.js - Offline Video Upload Mudra Processing and Dashboard Generator (Natyam)
document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("videoUploadForm");
    const fileInput = document.getElementById("videoFileInput");
    
    const progressContainer = document.getElementById("uploadProgressContainer");
    const progressText = document.getElementById("progressStatusText");
    const progressBar = document.getElementById("uploadProgressBar");
    const progressPercentage = document.getElementById("progressPercentage");
    const elapsedTimeDisplay = document.getElementById("elapsedTimeDisplay");
    
    // Results & Workspace Elements
    const uploadInitialState = document.getElementById("uploadInitialState");
    const uploadResultsState = document.getElementById("uploadResultsState");
    const videoPlaceholder = document.getElementById("processedVideoPlaceholder");
    const videoPlayer = document.getElementById("processedVideoPlayer");
    
    // Vector skeleton overlay canvas elements
    const canvas = document.getElementById("videoSkeletonCanvas");
    const ctx = canvas.getContext("2d");
    
    // Live Playback Metric elements
    const videoMudraName = document.getElementById("videoMudraName");
    const videoMudraStatus = document.getElementById("videoMudraStatus");
    const videoMudraConf = document.getElementById("videoMudraConf");
    const videoMudraBar = document.getElementById("videoMudraBar");
    
    const videoTimelineLog = document.getElementById("videoTimelineLog");
    const btnUploadAnother = document.getElementById("btnUploadAnother");

    // Strictly defined list of valid Bharatanatyam mudras
    const VALID_MUDRAS = [
        "chandrakala",
        "pataka",
        "mushti",
        "shikara",
        "mrigasirsha",
        "alapadma",
        "samyuta"
    ];

    // Playback state tracking
    let activeFrameLog = [];
    let elapsedTimerInterval = null;
    let elapsedSeconds = 0;

    // Vector drawing color design tokens
    const COLOR_GOLD = "#D4AF37";
    const COLOR_MAROON = "#6A0D25";

    function startElapsedTimer() {
        elapsedSeconds = 0;
        updateElapsedDisplay();
        elapsedTimerInterval = setInterval(() => {
            elapsedSeconds++;
            updateElapsedDisplay();
        }, 1000);
    }

    function stopElapsedTimer() {
        if (elapsedTimerInterval) {
            clearInterval(elapsedTimerInterval);
            elapsedTimerInterval = null;
        }
    }

    function updateElapsedDisplay() {
        if (!elapsedTimeDisplay) return;
        const mins = Math.floor(elapsedSeconds / 60);
        const secs = elapsedSeconds % 60;
        elapsedTimeDisplay.innerHTML = `<i class="fa-regular fa-clock me-1"></i> ${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Helper function to process video file on the client-side
    async function processVideoClientSide(file) {
        // Create an offscreen video element
        const tempVideo = document.createElement("video");
        tempVideo.muted = true;
        tempVideo.playsInline = true;
        tempVideo.preload = "auto";
        
        const fileUrl = URL.createObjectURL(file);
        tempVideo.src = fileUrl;
        
        // Initialize MediaPipe Hands inside this scope
        const videoHands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        
        videoHands.setOptions({
            maxNumHands: 1, // Single-hand optimization for processing speed
            modelComplexity: 1,
            minDetectionConfidence: 0.65,
            minTrackingConfidence: 0.5
        });

        let currentResults = null;
        videoHands.onResults((results) => {
            currentResults = results;
        });

        // Helper to seek to a timestamp and wait for seeked event
        const seekToTime = (time) => new Promise((resolve) => {
            const onSeeked = () => {
                tempVideo.removeEventListener("seeked", onSeeked);
                resolve();
            };
            tempVideo.addEventListener("seeked", onSeeked);
            tempVideo.currentTime = time;
            
            // Fallback timeout in case seek stalls
            setTimeout(() => {
                tempVideo.removeEventListener("seeked", onSeeked);
                resolve();
            }, 1000);
        });

        return new Promise((resolve, reject) => {
            tempVideo.onloadedmetadata = async () => {
                const duration = tempVideo.duration;
                if (!duration || isNaN(duration)) {
                    reject("Invalid video duration.");
                    return;
                }

                const interval = 0.125; // 8 frames/sec analysis
                let currentTime = 0;
                const frameLog = [];
                
                try {
                    while (currentTime <= duration) {
                        const percent = Math.min(98, Math.round((currentTime / duration) * 100));
                        progressBar.style.width = `${percent}%`;
                        progressPercentage.innerText = `${percent}%`;
                        progressText.innerHTML = `<i class="fa-solid fa-cog fa-spin me-2"></i> Analyzing frame at ${currentTime.toFixed(1)}s / ${duration.toFixed(1)}s...`;

                        await seekToTime(currentTime);

                        // Run MediaPipe hands model on current frame
                        currentResults = null;
                        await videoHands.send({ image: tempVideo });
                        
                        let mudra = "None";
                        let confidence = 0.0;
                        let handsData = [];

                        if (currentResults && currentResults.multiHandLandmarks && currentResults.multiHandLandmarks.length > 0) {
                            const hand = currentResults.multiHandLandmarks[0];
                            
                            // Normalised coordinates for the overlay
                            handsData.push(hand.map(lm => ({ x: lm.x, y: lm.y })));

                            // Prepare landmarks scaled to reference 640x480 frame for API
                            const xCoords = hand.map(lm => lm.x * 640);
                            const yCoords = hand.map(lm => lm.y * 480);
                            const landmarks42 = xCoords.concat(yCoords);

                            // Request mudra prediction from the server
                            try {
                                const response = await fetch("/api/predict-landmarks", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ landmarks: landmarks42 })
                                });
                                const predictData = await response.json();
                                if (predictData.mudra) {
                                    mudra = predictData.mudra;
                                    confidence = predictData.confidence;
                                }
                            } catch (err) {
                                console.error("Mudra predict API call failed at time:", currentTime, err);
                            }
                        }

                        frameLog.push({
                            timestamp: parseFloat(currentTime.toFixed(2)),
                            mudra: mudra,
                            mudra_confidence: confidence,
                            hands: handsData
                        });

                        currentTime += interval;
                    }

                    // Done processing!
                    progressBar.style.width = "100%";
                    progressPercentage.innerText = "100%";
                    progressText.innerHTML = `<i class="fa-solid fa-circle-check me-2"></i> Processing complete! Generating dashboard...`;
                    
                    videoHands.close();
                    URL.revokeObjectURL(fileUrl);
                    resolve({ frame_log: frameLog });

                } catch (err) {
                    videoHands.close();
                    URL.revokeObjectURL(fileUrl);
                    reject(err);
                }
            };

            tempVideo.onerror = (err) => {
                videoHands.close();
                URL.revokeObjectURL(fileUrl);
                reject("Failed to load video file. Ensure format is MP4, AVI, or MOV.");
            };
        });
    }

    // ----------------- SUBMIT HANDLER -----------------
    uploadForm.addEventListener("submit", (e) => {
        e.preventDefault();
        
        const files = fileInput.files;
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Reset and show progress bar
        progressContainer.classList.remove("d-none");
        progressBar.style.width = "0%";
        progressPercentage.innerText = "0%";
        progressText.innerHTML = `<i class="fa-solid fa-cog fa-spin me-2"></i> Initializing MediaPipe processor...`;
        
        // Start elapsed time counter
        startElapsedTimer();
        
        // Hide previous results and show workspace initial state
        uploadInitialState.classList.remove("d-none");
        uploadResultsState.classList.add("d-none");
        
        processVideoClientSide(file)
            .then((responseData) => {
                stopElapsedTimer();
                setTimeout(() => {
                    renderUploadDashboard(responseData, file);
                }, 500);
            })
            .catch((err) => {
                stopElapsedTimer();
                handleError(err);
            });
    });


    // ----------------- RESULTS RENDERER -----------------
    function renderUploadDashboard(data, file) {
        // 1. Swap initial state workspace for active playback workspace
        uploadInitialState.classList.add("d-none");
        uploadResultsState.classList.remove("d-none");
        
        // Clean up previous blob URL if any to prevent memory leaks
        if (videoPlayer.src && videoPlayer.src.startsWith("blob:")) {
            URL.revokeObjectURL(videoPlayer.src);
        }
        
        // Load the local video directly via a Blob URL to bypass browser codec compatibility issues
        videoPlayer.src = URL.createObjectURL(file);
        videoPlayer.load();
        
        // 2. Keep frame log reference
        activeFrameLog = data.frame_log || [];
        
        // 3. Reset Latch State (panel only — no external var needed)
        updateLivePanel("Waiting for Mudra...", 0.0, "Waiting");
        
        // 4. Bind dynamic size observer to the vector skeleton canvas overlay
        videoPlayer.addEventListener("loadedmetadata", () => {
            canvas.width = videoPlayer.clientWidth || videoPlayer.videoWidth;
            canvas.height = videoPlayer.clientHeight || videoPlayer.videoHeight;
        });
        
        // Handle window/container resizing dynamically
        const resizeObserver = new ResizeObserver(() => {
            canvas.width = videoPlayer.clientWidth;
            canvas.height = videoPlayer.clientHeight;
        });
        resizeObserver.observe(videoPlayer);
        
        // 5. Extract transitions and render timeline log (strictly valid only)
        const transitions = extractMudraTransitions(activeFrameLog);
        renderTimeline(transitions);
        
        // Reset Live prediction display
        updateLivePanel("Waiting for Mudra...", 0.0, "Waiting");
        
        // Hide progress loader after a short delay
        setTimeout(() => {
            progressContainer.classList.add("d-none");
        }, 1500);
    }

    // ----------------- TRANSITION EXTRACTION -----------------
    function extractMudraTransitions(frameLog) {
        const transitions = [];
        let lastMudra = null;
        
        frameLog.forEach((frame) => {
            let currentMudra = frame.mudra || "None";
            const mudraLabel = currentMudra.trim().toLowerCase();
            const confidence = frame.mudra_confidence || 0;
            
            // Only capture predictions that belong strictly to the 7 valid mudras with confidence >= 0.85
            if (VALID_MUDRAS.includes(mudraLabel) && confidence >= 0.85) {
                // If the same mudra continues, display it only once (transition tracking)
                if (mudraLabel !== lastMudra) {
                    // Convert back to Title Case for visual representation
                    const titleMudra = mudraLabel.charAt(0).toUpperCase() + mudraLabel.slice(1);
                    transitions.push({
                        timestamp: frame.timestamp,
                        mudra: titleMudra,
                        confidence: confidence
                    });
                    lastMudra = mudraLabel;
                }
            }
        });
        
        return transitions;
    }

    // ----------------- TIMELINE RENDERING -----------------
    function renderTimeline(transitions) {
        videoTimelineLog.innerHTML = "";
        
        if (transitions.length === 0) {
            videoTimelineLog.innerHTML = `<li class="list-group-item text-center text-muted py-3">No valid mudras detected</li>`;
            return;
        }
        
        transitions.forEach((trans) => {
            const timeStr = formatTime(trans.timestamp);
            const confPct = Math.round(trans.confidence * 100);
            
            const li = document.createElement("li");
            li.className = "list-group-item bg-transparent border-0 py-2 timeline-item-bharata";
            li.setAttribute("data-time", trans.timestamp);
            
            li.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <span class="text-muted fw-bold me-2" style="font-family: monospace;">${timeStr}</span>
                        <span class="text-uppercase fw-semibold" style="color: var(--deep-maroon);">${trans.mudra}</span>
                    </div>
                    <span class="badge bg-success-subtle text-success border border-light-subtle">${confPct}%</span>
                </div>
            `;
            
            // Seek on click
            li.addEventListener("click", () => {
                videoPlayer.currentTime = trans.timestamp;
                videoPlayer.play();
                
                // Highlight active item
                document.querySelectorAll(".timeline-item-bharata").forEach(item => {
                    item.classList.remove("active");
                });
                li.classList.add("active");
            });
            
            videoTimelineLog.appendChild(li);
        });
    }

    // ----------------- VECTOR DRAWING HELPERS -----------------
    function drawLine(p1, p2, color, thickness) {
        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.stroke();
    }

    function drawHandSkeleton(landmarks, color) {
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12],
            [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [17, 18], [18, 19], [19, 20],
            [0, 17]
        ];
        
        // Connections
        connections.forEach(([p1, p2]) => {
            drawLine(landmarks[p1], landmarks[p2], color, 2.5);
        });

        // Joints
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

    // ----------------- VIDEO STATE SYNCING -----------------
    videoPlayer.addEventListener("timeupdate", () => {
        const currentTime = videoPlayer.currentTime;
        
        // Clear canvas first on every playback tick
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (activeFrameLog.length > 0) {
            // Binary search for the closest log entry — O(log n) vs O(n)
            let lo = 0, hi = activeFrameLog.length - 1, mid;
            while (lo < hi) {
                mid = (lo + hi) >> 1;
                if (activeFrameLog[mid].timestamp < currentTime) lo = mid + 1;
                else hi = mid;
            }
            // Pick the closer of lo-1 and lo
            const closestFrame =
                (lo > 0 &&
                 Math.abs(activeFrameLog[lo - 1].timestamp - currentTime) <
                 Math.abs(activeFrameLog[lo].timestamp - currentTime))
                    ? activeFrameLog[lo - 1]
                    : activeFrameLog[lo];
            
            // Draw vector hand landmarks skeleton and bounding box overlays
            if (closestFrame.hands && closestFrame.hands.length > 0) {
                closestFrame.hands.forEach(handLandmarks => {
                    // Draw joints & connections
                    drawHandSkeleton(handLandmarks, COLOR_MAROON);
                    
                    // Draw bounding box
                    const xs = handLandmarks.map(lm => lm.x * canvas.width);
                    const ys = handLandmarks.map(lm => lm.y * canvas.height);
                    const xMin = Math.max(0, Math.min(...xs) - 15);
                    const xMax = Math.min(canvas.width, Math.max(...xs) + 15);
                    const yMin = Math.max(0, Math.min(...ys) - 15);
                    const yMax = Math.min(canvas.height, Math.max(...ys) + 15);
                    
                    ctx.strokeStyle = COLOR_GOLD;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
                });
            }
            
            // Sync current mudra display panel (Real-time frame prediction lookup)
            const detectedMudra = (closestFrame.mudra || "").trim().toLowerCase();
            const confidence = closestFrame.mudra_confidence || 0.0;
            const hasHand = closestFrame.hands && closestFrame.hands.length > 0;
            
            if (!hasHand || detectedMudra === "none") {
                updateLivePanel("No Hand Detected", 0.0, "Waiting");
            } else if (detectedMudra === "unrecognized") {
                updateLivePanel("Unrecognized", confidence, "Low Confidence");
            } else if (VALID_MUDRAS.includes(detectedMudra)) {
                const formattedName = detectedMudra.charAt(0).toUpperCase() + detectedMudra.slice(1);
                const status = confidence >= 0.85 ? "Recognized" : "Low Confidence";
                updateLivePanel(formattedName, confidence, status);
            } else {
                const formattedName = detectedMudra.charAt(0).toUpperCase() + detectedMudra.slice(1);
                const status = confidence >= 0.85 ? "Recognized" : "Low Confidence";
                updateLivePanel(formattedName, confidence, status);
            }
            
            // Auto scroll & highlight active timeline items
            highlightActiveTimelineItem(currentTime);
        }
    });

    function updateLivePanel(mudra, confidence, status) {
        if (!videoMudraName) return;
        
        const confPct = Math.round(confidence * 100);
        videoMudraName.innerText = mudra;
        
        if (videoMudraStatus) {
            videoMudraStatus.innerText = status || "Waiting...";
            if (status === "Recognized") {
                videoMudraStatus.className = "badge bg-success px-3 py-2 rounded-pill fs-6";
            } else if (status === "Low Confidence") {
                videoMudraStatus.className = "badge bg-danger px-3 py-2 rounded-pill fs-6";
            } else {
                videoMudraStatus.className = "badge bg-secondary px-3 py-2 rounded-pill fs-6";
            }
        }
        
        if (videoMudraConf) {
            videoMudraConf.innerText = `${confPct}%`;
        }
        if (videoMudraBar) {
            videoMudraBar.style.width = `${confPct}%`;
        }
    }

    function highlightActiveTimelineItem(currentTime) {
        const items = document.querySelectorAll(".timeline-item-bharata");
        if (items.length === 0) return;
        
        let activeIndex = -1;
        
        // Find closest transition item that started at or before the current playback time
        for (let i = 0; i < items.length; i++) {
            const itemTime = parseFloat(items[i].getAttribute("data-time"));
            if (itemTime <= currentTime) {
                activeIndex = i;
            } else {
                break;
            }
        }
        
        if (activeIndex !== -1) {
            items.forEach((item, idx) => {
                if (idx === activeIndex) {
                    if (!item.classList.contains("active")) {
                        item.classList.add("active");
                        
                        // Scroll smoothly inside timeline container
                        const container = document.getElementById("videoTimelineContainer");
                        const itemTop = item.offsetTop - container.offsetTop;
                        container.scrollTo({
                            top: itemTop - container.clientHeight / 2 + item.clientHeight / 2,
                            behavior: "smooth"
                        });
                    }
                } else {
                    item.classList.remove("active");
                }
            });
        }
    }

    // ----------------- RESET / UPLOAD ANOTHER -----------------
    if (btnUploadAnother) {
        btnUploadAnother.addEventListener("click", () => {
            uploadForm.reset();
            
            // Restore workspace initial state
            uploadInitialState.classList.remove("d-none");
            uploadResultsState.classList.add("d-none");
            
            progressContainer.classList.add("d-none");
            progressBar.style.width = "0%";
            progressPercentage.innerText = "0%";
            
            // Stop and flush player & canvas
            videoPlayer.pause();
            if (videoPlayer.src && videoPlayer.src.startsWith("blob:")) {
                URL.revokeObjectURL(videoPlayer.src);
            }
            videoPlayer.src = "";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            activeFrameLog = [];
            updateLivePanel("Waiting for Mudra...", 0.0, "Waiting");
        });
    }

    // ----------------- TIME FORMATTING HELPERS -----------------
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function handleError(message) {
        stopElapsedTimer();
        progressBar.style.width = "0%";
        progressPercentage.innerText = "0%";
        progressText.innerHTML = `<span class="text-danger"><i class="fa-solid fa-circle-exclamation me-2"></i> ${message}</span>`;
        if (window.progressInterval) {
            clearInterval(window.progressInterval);
        }
        alert(`Error: ${message}`);
    }
});
