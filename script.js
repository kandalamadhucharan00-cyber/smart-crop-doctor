document.addEventListener('DOMContentLoaded', () => {
    
    // --- Tabs Navigation ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const sections = document.querySelectorAll('.view-section');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            tabBtns.forEach(t => t.classList.remove('active'));
            sections.forEach(s => s.classList.add('hidden'));
            
            // Add active to clicked tab
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.remove('hidden');

            // Handle Camera logic safely
            if (targetId === 'camera-view') {
                // Do not auto start to save resources, wait for user click
            } else {
                stopCamera();
            }

            if (targetId === 'history-view') {
                loadHistory();
            }
        });
    });

    // --- Language Logic ---
    let currentAnalysisData = null;
    const langSelect = document.getElementById('language-select');
    
    langSelect.addEventListener('change', () => {
        if (currentAnalysisData) {
            updateTextContent();
        }
    });

    function updateTextContent() {
        if (!currentAnalysisData) return;
        const lang = langSelect.value;
        const info = currentAnalysisData.disease_info;
        
        document.getElementById('val-disease').innerText = info.name[lang] || info.name['en'];
        document.getElementById('val-description').innerText = info.description[lang] || info.description['en'];
        document.getElementById('val-recommendation').innerText = info.recommendation[lang] || info.recommendation['en'];
        
        const labels = {
            'en': { desc: 'Description', treat: 'Treatment Plan' },
            'te': { desc: 'వివరణ', treat: 'చికిత్స ప్రణాళిక' },
            'kn': { desc: 'ವಿವರಣೆ', treat: 'ಚಿಕಿತ್ಸಾ ಯೋಜನೆ' }
        };
        document.getElementById('label-desc').innerText = labels[lang].desc;
        document.getElementById('label-treat').innerText = labels[lang].treat;
    }

    // --- Upload Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const resultsPanel = document.getElementById('results-panel');
    const resetBtn = document.getElementById('reset-btn');

    // Trigger file input
    browseBtn.addEventListener('click', () => fileInput.click());

    // Drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // Drop handler
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if(files.length > 0) handleFile(files[0]);
    });

    // File input handler
    fileInput.addEventListener('change', function() {
        if(this.files.length > 0) handleFile(this.files[0]);
    });

    resetBtn.addEventListener('click', () => {
        resultsPanel.classList.add('hidden');
        dropZone.style.display = 'block';
        fileInput.value = '';
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert("Please upload an image file.");
            return;
        }

        dropZone.style.display = 'none';
        resultsPanel.classList.remove('hidden');
        resultsPanel.classList.add('scanning'); // add scan animation

        // Build UI state to processing
        document.getElementById('result-img').src = URL.createObjectURL(file);
        document.getElementById('val-disease').innerText = "Processing...";
        document.getElementById('conf-progress').style.width = "0%";
        document.getElementById('val-conf').innerText = "0%";
        document.getElementById('val-severity').innerText = "Pending";
        document.getElementById('val-severity').className = "badge";
        document.getElementById('val-description').innerText = "Awaiting analysis...";
        document.getElementById('val-recommendation').innerText = "Awaiting analysis...";

        // Prepare File data
        const formData = new FormData();
        formData.append('file', file);

        // Fetch API
        fetch('/predict/image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            resultsPanel.classList.remove('scanning');
            if (data.error) {
                alert("Error: " + data.error);
                resetBtn.click();
                return;
            }
            updateResultsUI(data);
        })
        .catch(error => {
            resultsPanel.classList.remove('scanning');
            console.error('Error:', error);
            alert("Failed to analyze image.");
            resetBtn.click();
        });
    }

    function updateResultsUI(data) {
        currentAnalysisData = data;
        
        // Update Image
        if (data.image) {
            document.getElementById('result-img').src = "data:image/jpeg;base64," + data.image;
        }
        
        // Update Info text based on current language
        updateTextContent();
        
        document.getElementById('conf-progress').style.width = data.confidence + "%";
        
        let confColor = 'var(--safe)';
        if(data.confidence < 50) confColor = 'var(--danger)';
        else if(data.confidence < 80) confColor = 'var(--warning)';
        document.getElementById('conf-progress').style.background = confColor;
        document.getElementById('val-conf').innerText = data.confidence + "%";

        const sevElem = document.getElementById('val-severity');
        sevElem.innerText = data.severity;
        sevElem.className = `badge severity-${data.severity}`;
    }

    // --- Camera Logic ---
    const cameraBtn = document.getElementById('toggle-camera-btn');
    const videoStream = document.getElementById('video-stream');
    let cameraActive = false;

    cameraBtn.addEventListener('click', () => {
        if(!cameraActive) {
            // Start Stream
            videoStream.src = "/video_feed";
            cameraBtn.innerText = "Stop Camera";
            cameraBtn.classList.replace('btn-primary', 'btn-outline');
            cameraActive = true;
        } else {
            stopCamera();
        }
    });

    function stopCamera() {
        if(cameraActive) {
            videoStream.src = ""; // Stops the mjpeg request
            cameraBtn.innerText = "Start Camera";
            cameraBtn.classList.replace('btn-outline', 'btn-primary');
            cameraActive = false;
        }
    }

    // --- History Logic ---
    function loadHistory() {
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;">Loading...</td></tr>';
        
        fetch('/api/history')
        .then(res => res.json())
        .then(data => {
            tbody.innerHTML = '';
            const history = data.history.reverse(); // Latest first
            if(history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;">No history found. Process an image first.</td></tr>';
                return;
            }

            history.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${item['Timestamp']}</td>
                    <td><strong>${item['Disease Detected']}</strong></td>
                    <td>${item['Confidence (%)']}%</td>
                    <td><span class="badge severity-${item['Severity']}">${item['Severity']}</span></td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => {
            console.error("Failed to load history", err);
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--danger);">Failed to load history</td></tr>';
        });
    }
});
