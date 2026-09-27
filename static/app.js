(() => {
    'use strict';

    const page = document.body.dataset.page;
    if (page !== 'home') return;

    const $ = (selector) => document.querySelector(selector);
    const imageInput = $('#image-input');
    const dropzone = $('#dropzone');
    const form = $('#prediction-form');
    const analyzeButton = $('#analyze-button');
    const selectedFile = $('#selected-file');
    const inputPreview = $('#input-preview');
    const logWindow = $('#log-window');
    const logEmpty = $('#log-empty');
    const resultSection = $('#result-section');
    const loader = $('#analysis-loader');
    const loaderTitle = $('#loader-title');
    const loaderCopy = $('#loader-copy');
    const loaderFill = $('#loader-track-fill');
    const seenEvents = new Set();
    let chosenFile = null;
    let previewUrl = null;
    let sourceStatus = 'loading';

    const loaderStages = {
        input: { title: 'Preparing image', copy: 'Validating the uploaded file and reading its dimensions.', progress: 22 },
        preprocess: { title: 'Preparing image', copy: 'Resizing and normalizing the RGB input.', progress: 40 },
        inference: { title: 'Running the model', copy: 'EfficientNet-B0 is scoring the three dataset labels.', progress: 68 },
        explanation: { title: 'Building explanation', copy: 'Generating a Grad-CAM view of the selected class score.', progress: 88 },
        complete: { title: 'Analysis complete', copy: 'The result and visual explanation are ready.', progress: 100 },
    };

    function setStatus(status, model = {}) {
        sourceStatus = status;
        const topStatus = $('#top-status');
        const topStatusLabel = $('#top-status-label');
        const stateTitle = $('#runtime-state-title');
        const stateCopy = $('#runtime-state-copy');
        const statusText = status === 'ready' ? 'model ready' : status === 'error' ? 'model error' : 'loading model';
        topStatus.dataset.status = status;
        topStatusLabel.textContent = statusText;
        const modelLabel = model.model_label || prettyModelName(model.model_name);
        $('#runtime-model').textContent = modelLabel;
        if (analyzeButton) analyzeButton.disabled = !chosenFile || status !== 'ready';
        $('#runtime-input').textContent = model.image_size ? `RGB / ${model.image_size} × ${model.image_size}` : 'RGB / 224 × 224';
        if (status === 'ready') {
            stateTitle.textContent = 'Model is ready';
            stateCopy.textContent = `${modelLabel} is loaded and waiting for one image.`;
        } else if (status === 'error') {
            stateTitle.textContent = 'Model needs attention';
            stateCopy.textContent = 'The exported model contract could not be loaded.';
        } else {
            stateTitle.textContent = 'Preparing model';
            stateCopy.textContent = 'Connecting to the local inference service…';
        }
    }

    function prettyModelName(name) {
        if (name === 'efficientnet_b0') return 'EfficientNet-B0';
        if (name === 'custom_cnn') return 'Custom CNN';
        return name ? name.replaceAll('_', ' ') : 'EfficientNet-B0';
    }

    function appendLog(event) {
        if (!event || seenEvents.has(event.id)) return;
        seenEvents.add(event.id);
        if (logEmpty) logEmpty.remove();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.dataset.level = event.level || 'info';
        entry.dataset.eventId = event.id;
        entry.innerHTML = `<span class="log-entry-time">${escapeHtml(event.time || '--:--:--')}</span><span class="log-entry-dot"></span><span>${escapeHtml(event.message || '')}</span>`;
        logWindow.appendChild(entry);
        while (logWindow.children.length > 38) logWindow.firstElementChild.remove();
        logWindow.scrollTop = logWindow.scrollHeight;
        if (event.kind === 'preprocess' || event.kind === 'inference' || event.kind === 'explanation') setLoaderStage(event.kind);
    }

    function escapeHtml(value) {
        return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));
    }

    function setLoaderStage(stage) {
        const details = loaderStages[stage] || loaderStages.input;
        loaderTitle.textContent = details.title;
        loaderCopy.textContent = details.copy;
        loaderFill.style.width = `${details.progress}%`;
        document.querySelectorAll('[data-loader-step]').forEach((item) => {
            item.classList.toggle('is-active', item.dataset.loaderStep === (stage === 'preprocess' ? 'input' : stage === 'inference' ? 'model' : stage === 'explanation' || stage === 'complete' ? 'explanation' : 'input'));
        });
    }

    function setLoading(isLoading) {
        loader.hidden = !isLoading;
        analyzeButton.disabled = isLoading || !chosenFile || sourceStatus !== 'ready';
        if (isLoading) setLoaderStage('input');
    }

    function displayFile(file) {
        if (!file) return;
        const allowed = ['image/jpeg', 'image/png', 'image/webp'];
        if (!allowed.includes(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
            showClientError('Choose a JPG, PNG, or WEBP image.');
            return;
        }
        chosenFile = file;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        inputPreview.src = previewUrl;
        $('#selected-file-name').textContent = file.name;
        $('#selected-file-meta').textContent = `${formatBytes(file.size)} · ${file.type || 'image'}`;
        selectedFile.hidden = false;
        dropzone.hidden = true;
        analyzeButton.disabled = sourceStatus !== 'ready';
    }

    function clearFile() {
        chosenFile = null;
        imageInput.value = '';
        selectedFile.hidden = true;
        dropzone.hidden = false;
        analyzeButton.disabled = true;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
        inputPreview.removeAttribute('src');
    }

    function formatBytes(bytes) {
        if (!bytes) return '0 B';
        const units = ['B', 'KB', 'MB'];
        const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
    }

    function showClientError(message) {
        appendLog({ id: `client-${Date.now()}`, time: new Date().toLocaleTimeString(), level: 'error', kind: 'error', message });
    }

    function renderProbabilities(probabilities) {
        const list = $('#probability-list');
        list.innerHTML = '';
        Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]).forEach(([label, probability]) => {
            const row = document.createElement('div');
            row.className = 'probability-row';
            row.innerHTML = `<div class="probability-top"><strong>${escapeHtml(label)}</strong><span>${(probability * 100).toFixed(2)}%</span></div><div class="probability-track"><div class="probability-fill" style="width: 0%"></div></div>`;
            list.appendChild(row);
            requestAnimationFrame(() => { row.querySelector('.probability-fill').style.width = `${probability * 100}%`; });
        });
    }

    function renderResult(result) {
        setLoaderStage('complete');
        $('#result-label').textContent = result.prediction || '—';
        $('#result-score').textContent = `${((result.confidence || 0) * 100).toFixed(2)}%`;
        $('#result-model').textContent = result.model_label || prettyModelName(result.model);
        $('#result-duration').textContent = `${result.duration_ms || '—'} ms · ${result.device || 'local'}`;
        $('#result-image-meta').textContent = result.image_meta ? `${result.image_meta.format} · ${result.image_meta.width}×${result.image_meta.height}` : 'uploaded image';
        $('#result-input-image').src = previewUrl || '';
        $('#result-explanation-image').src = result.explanation_image || '';
        $('#explanation-title').textContent = `${result.prediction || 'Label'} / interpretation`;
        $('#explanation-context').textContent = result.explanation?.label_context || 'The model selected the highest-scoring dataset label.';
        $('#explanation-method').textContent = `${result.explanation?.method || 'Grad-CAM'} — ${result.explanation?.method_note || ''}`;
        $('#explanation-score-note').textContent = result.explanation?.score_note || '';
        $('#result-notice').textContent = result.notice || '';
        renderProbabilities(result.probabilities);
        resultSection.hidden = false;
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    async function fetchStatus() {
        try {
            const response = await fetch('/api/status', { headers: { Accept: 'application/json' } });
            const status = await response.json();
            setStatus(status.status, status.model || {});
            (status.events || []).forEach(appendLog);
        } catch (_error) {
            setStatus('error');
            showClientError('Could not connect to the local status endpoint.');
        }
    }

    function connectEvents() {
        if (!window.EventSource) {
            $('#connection-state').innerHTML = '<span class="status-dot"></span> status fallback';
            return;
        }
        const eventSource = new EventSource('/events');
        eventSource.onopen = () => {
            $('#connection-state').classList.add('is-connected');
            $('#connection-state').innerHTML = '<span class="status-dot"></span> live stream';
        };
        eventSource.onmessage = (message) => {
            try {
                const event = JSON.parse(message.data);
                appendLog(event);
                if (event.kind === 'model' && /ready/i.test(event.message)) {
                    setStatus('ready', { model_label: $('#runtime-model').textContent, image_size: 224 });
                }
                if (event.level === 'error') setStatus('error');
            } catch (_error) { /* ignore malformed keep-alive data */ }
        };
        eventSource.onerror = () => {
            $('#connection-state').classList.remove('is-connected');
            $('#connection-state').innerHTML = '<span class="status-dot"></span> reconnecting';
        };
    }

    dropzone.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', () => displayFile(imageInput.files[0]));
    $('#remove-file').addEventListener('click', clearFile);
    ['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('is-dragging'); }));
    ['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('is-dragging'); }));
    dropzone.addEventListener('drop', (event) => displayFile(event.dataTransfer.files[0]));

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!chosenFile) return showClientError('Choose an image before running analysis.');
        setLoading(true);
        const payload = new FormData();
        payload.append('image', chosenFile, chosenFile.name);
        appendLog({ id: `client-submit-${Date.now()}`, time: new Date().toLocaleTimeString(), kind: 'input', level: 'info', message: 'Request sent to the local inference service.' });
        try {
            const response = await fetch('/predict?format=json', { method: 'POST', body: payload, headers: { Accept: 'application/json' } });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Analysis failed.');
            renderResult(result);
        } catch (error) {
            showClientError(error.message);
        } finally {
            setLoading(false);
        }
    });

    $('#new-analysis').addEventListener('click', () => {
        resultSection.hidden = true;
        clearFile();
        window.scrollTo({ top: document.querySelector('.workspace').offsetTop - 20, behavior: 'smooth' });
    });

    const initialResult = $('#initial-result');
    if (initialResult) {
        try { renderResult(JSON.parse(initialResult.textContent)); } catch (_error) { /* no-op */ }
    }
    fetchStatus();
    connectEvents();
})();
