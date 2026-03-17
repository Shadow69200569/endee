// ============================================================
//   Endee Vision – Frontend Logic
// ============================================================

const API_BASE = window.location.origin;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const errorMessage = document.getElementById('errorMessage');
const serverStatus = document.getElementById('serverStatus');
const latencyBadge = document.getElementById('latencyBadge');

let selectedFile = null;

// ---- Server status check ----
async function checkHealth() {
    const dot = serverStatus.querySelector('.status-dot');
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        const data = await res.json();
        dot.className = 'status-dot online';
        serverStatus.innerHTML = `<span class="status-dot online"></span> Server online`;
        if (data.db_connected) {
            serverStatus.innerHTML = `<span class="status-dot online"></span> Server & DB online`;
        }
    } catch {
        dot.className = 'status-dot offline';
        serverStatus.innerHTML = `<span class="status-dot offline"></span> Server offline`;
    }
}
checkHealth();
setInterval(checkHealth, 15000);

// ---- File selection ----
dropZone.addEventListener('click', (e) => {
    if (!previewArea.classList.contains('hidden')) return;
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) loadFile(e.target.files[0]);
});

// ---- Drag & Drop ----
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadFile(file);
});

function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewArea.classList.remove('hidden');
        // hide the static upload content
        dropZone.querySelector('.upload-content').style.display = 'none';
        // Clear old results
        hideResults();
        hideError();
    };
    reader.readAsDataURL(file);
}

// ---- Clear / Reset ----
clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    reset();
});

function reset() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewArea.classList.add('hidden');
    dropZone.querySelector('.upload-content').style.display = '';
    hideResults();
    hideError();
}

// ---- Search ----
searchBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!selectedFile) return;
    await runSearch(selectedFile);
});

async function runSearch(file) {
    showLoader();
    hideResults();
    hideError();

    const formData = new FormData();
    formData.append('file', file);

    const startTime = Date.now();
    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        hideLoader();

        if (!data.results || data.results.length === 0) {
            showError('No similar images found. Make sure you have run the embedding generation script. See the README for setup instructions.');
            return;
        }

        showResults(data.results, elapsed);
    } catch (err) {
        hideLoader();
        showError(`Search failed: ${err.message}`);
    }
}

// ---- Rendering ----
function showResults(results, elapsed) {
    latencyBadge.textContent = `Retrieved in ${elapsed}s`;
    resultsGrid.innerHTML = '';

    results.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'result-card';

        // Convert cosine similarity distance to percentage (distance closer to 0 = more similar)
        const similarity = Math.max(0, Math.min(100, (1 - item.distance) * 100));
        const simLabel = similarity.toFixed(1) + '%';

        card.innerHTML = `
            <img src="${item.path || ''}" alt="${item.id}"
                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22240%22 height=%22200%22 viewBox=%220 0 240 200%22><rect fill=%22%2318181f%22 width=%22240%22 height=%22200%22/><text x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%23556%22 font-size=%2214%22>Image not found</text></svg>'">
            <div class="card-info">
                <div class="card-id" title="${item.id}">${item.id}</div>
                <div class="similarity-bar">
                    <div class="bar-track"><div class="bar-fill" style="width:${similarity.toFixed(1)}%"></div></div>
                    <span class="similarity-label">${simLabel}</span>
                </div>
            </div>
        `;
        resultsGrid.appendChild(card);
    });

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Helpers ----
function showLoader() { loader.classList.remove('hidden'); }
function hideLoader() { loader.classList.add('hidden'); }
function hideResults() { resultsSection.classList.add('hidden'); }
function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}
function hideError() { errorMessage.classList.add('hidden'); }
