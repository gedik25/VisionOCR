/**
 * VisionOCR Web — Interactive Single Canvas & Batch Processing Dashboard Logic
 * Built according to apple-design, emil-design-eng & impeccable principles
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── State ──
  const state = {
    mode: 'single', // 'single' | 'batch'
    currentFile: null,
    currentImageUrl: null,
    imageWidth: 0,
    imageHeight: 0,
    zoom: 1.0,
    panX: 0,
    panY: 0,
    rotation: 0,
    isPanning: false,
    startX: 0,
    startY: 0,
    lastResult: null,
    isProcessing: false,
    activeStreamReader: null,

    // Batch State
    batchFiles: [],
    batchResults: [],
    lastBatchData: null,
    isBatchProcessing: false,
    batchStartTime: 0
  };

  // ── DOM References (Navigation & Controls) ──
  const btnModeSingle = document.getElementById('btnModeSingle');
  const btnModeBatch = document.getElementById('btnModeBatch');
  const singleWorkspace = document.getElementById('singleWorkspace');
  const batchWorkspace = document.getElementById('batchWorkspace');
  const singleHeaderActions = document.getElementById('singleHeaderActions');
  const modelSelect = document.getElementById('modelSelect');
  const hardwareBadge = document.getElementById('hardwareBadge');
  const hardwareText = document.getElementById('hardwareText');

  // Single Mode DOM
  const fileInput = document.getElementById('fileInput');
  const btnSelectImage = document.getElementById('btnSelectImage');
  const btnRunOcr = document.getElementById('btnRunOcr');
  const btnRunText = document.getElementById('btnRunText');

  const canvasViewport = document.getElementById('canvasViewport');
  const emptyDropzone = document.getElementById('emptyDropzone');
  const canvasTransformWrapper = document.getElementById('canvasTransformWrapper');
  const sourceImage = document.getElementById('sourceImage');
  const boundingBoxSvg = document.getElementById('boundingBoxSvg');
  const floatingToolbar = document.getElementById('floatingToolbar');
  const zoomIndicator = document.getElementById('zoomIndicator');

  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnFit = document.getElementById('btnFit');
  const btnActualSize = document.getElementById('btnActualSize');
  const btnRotate = document.getElementById('btnRotate');
  const btnReset = document.getElementById('btnReset');

  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  const markdownOutput = document.getElementById('markdownOutput');
  const plainTextOutput = document.getElementById('plainTextOutput');
  const jsonOutput = document.getElementById('jsonOutput');
  const statModel = document.getElementById('statModel');
  const statLatency = document.getElementById('statLatency');
  const statDevice = document.getElementById('statDevice');
  const statWords = document.getElementById('statWords');
  const statChars = document.getElementById('statChars');
  const statBoxes = document.getElementById('statBoxes');

  const btnCopyText = document.getElementById('btnCopyText');
  const btnExportTxt = document.getElementById('btnExportTxt');
  const btnExportJson = document.getElementById('btnExportJson');
  const btnExportMd = document.getElementById('btnExportMd');
  const toastContainer = document.getElementById('toastContainer');

  // Batch Mode DOM
  const batchDropzone = document.getElementById('batchDropzone');
  const batchFolderInput = document.getElementById('batchFolderInput');
  const batchFilesInput = document.getElementById('batchFilesInput');
  const btnSelectBatchFolder = document.getElementById('btnSelectBatchFolder');
  const btnSelectBatchFiles = document.getElementById('btnSelectBatchFiles');
  const batchFileCountText = document.getElementById('batchFileCountText');
  const batchFormatSelect = document.getElementById('batchFormatSelect');
  const btnRunBatch = document.getElementById('btnRunBatch');
  const btnRunBatchText = document.getElementById('btnRunBatchText');
  const batchFileChips = document.getElementById('batchFileChips');

  const batchProgressHeading = document.getElementById('batchProgressHeading');
  const batchProgressCounter = document.getElementById('batchProgressCounter');
  const batchProgressFill = document.getElementById('batchProgressFill');
  const progressPulse = document.getElementById('progressPulse');

  const batchStatTotal = document.getElementById('batchStatTotal');
  const batchStatSuccess = document.getElementById('batchStatSuccess');
  const batchStatFailed = document.getElementById('batchStatFailed');
  const batchStatTime = document.getElementById('batchStatTime');

  const batchActiveItemCard = document.getElementById('batchActiveItemCard');
  const batchActiveFilename = document.getElementById('batchActiveFilename');
  const batchActiveLatency = document.getElementById('batchActiveLatency');
  const batchActivePreview = document.getElementById('batchActivePreview');

  const batchExportToolbar = document.getElementById('batchExportToolbar');
  const batchSummaryText = document.getElementById('batchSummaryText');
  const btnBatchDownloadCsv = document.getElementById('btnBatchDownloadCsv');
  const btnBatchDownloadJson = document.getElementById('btnBatchDownloadJson');
  const btnBatchDownloadZip = document.getElementById('btnBatchDownloadZip');

  const batchTableSearch = document.getElementById('batchTableSearch');
  const batchTableBody = document.getElementById('batchTableBody');

  // ── Helper: Safe HTML Escaping ──
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ── 1. Toast Notification System (Sonner Style) ──
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' : ''}`;
    toast.innerHTML = `
      <span>${type === 'error' ? '⚠️' : '✓'}</span>
      <span>${escapeHtml(message)}</span>
    `;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = 'toastOut 200ms var(--ease-spring) forwards';
      setTimeout(() => toast.remove(), 200);
    }, 2800);
  }

  // ── 2. Mode Switching (Single vs Batch) ──
  function switchMode(newMode) {
    state.mode = newMode;
    if (newMode === 'single') {
      btnModeSingle.classList.add('active');
      btnModeBatch.classList.remove('active');
      singleWorkspace.style.display = 'grid';
      batchWorkspace.style.display = 'none';
      if (singleHeaderActions) singleHeaderActions.style.display = 'flex';
    } else {
      btnModeBatch.classList.add('active');
      btnModeSingle.classList.remove('active');
      singleWorkspace.style.display = 'none';
      batchWorkspace.style.display = 'grid';
      if (singleHeaderActions) singleHeaderActions.style.display = 'none';
    }
  }

  btnModeSingle.addEventListener('click', () => switchMode('single'));
  btnModeBatch.addEventListener('click', () => switchMode('batch'));

  // ── 3. Single Mode Canvas & Transform Logic ──
  function handleImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      showToast('Lütfen geçerli bir görsel dosyası seçin.', 'error');
      return;
    }

    state.currentFile = file;
    const url = URL.createObjectURL(file);
    state.currentImageUrl = url;

    sourceImage.onload = () => {
      state.imageWidth = sourceImage.naturalWidth;
      state.imageHeight = sourceImage.naturalHeight;
      
      emptyDropzone.style.display = 'none';
      canvasTransformWrapper.style.display = 'block';
      floatingToolbar.style.display = 'flex';

      state.rotation = 0;
      clearBoundingBoxes();
      fitToViewport();
      showToast(`Görsel yüklendi: ${file.name} (${state.imageWidth}×${state.imageHeight})`);
    };

    sourceImage.src = url;
  }

  function updateTransform() {
    canvasTransformWrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom}) rotate(${state.rotation}deg)`;
    zoomIndicator.textContent = `${Math.round(state.zoom * 100)}%`;
  }

  function fitToViewport() {
    if (!state.imageWidth || !state.imageHeight) return;

    const vpRect = canvasViewport.getBoundingClientRect();
    const vpW = vpRect.width - 48;
    const vpH = vpRect.height - 48;

    const scaleX = vpW / state.imageWidth;
    const scaleY = vpH / state.imageHeight;
    const fitScale = Math.min(scaleX, scaleY, 1.0);

    state.zoom = Math.max(fitScale, 0.05);

    const renderedW = state.imageWidth * state.zoom;
    const renderedH = state.imageHeight * state.zoom;
    state.panX = (vpRect.width - renderedW) / 2;
    state.panY = (vpRect.height - renderedH) / 2;

    updateTransform();
  }

  function setActualSize() {
    if (!state.imageWidth || !state.imageHeight) return;
    const vpRect = canvasViewport.getBoundingClientRect();
    state.zoom = 1.0;
    state.panX = (vpRect.width - state.imageWidth) / 2;
    state.panY = (vpRect.height - state.imageHeight) / 2;
    updateTransform();
  }

  canvasViewport.addEventListener('wheel', (e) => {
    if (!state.currentFile) return;
    e.preventDefault();

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.min(Math.max(state.zoom * zoomFactor, 0.05), 25.0);

    const vpRect = canvasViewport.getBoundingClientRect();
    const cursorX = e.clientX - vpRect.left;
    const cursorY = e.clientY - vpRect.top;

    state.panX = cursorX - (cursorX - state.panX) * (newZoom / state.zoom);
    state.panY = cursorY - (cursorY - state.panY) * (newZoom / state.zoom);
    state.zoom = newZoom;

    updateTransform();
  }, { passive: false });

  canvasViewport.addEventListener('pointerdown', (e) => {
    if (!state.currentFile || e.target.closest('.floating-toolbar')) return;
    state.isPanning = true;
    state.startX = e.clientX - state.panX;
    state.startY = e.clientY - state.panY;
    canvasViewport.classList.add('panning');
    canvasViewport.setPointerCapture(e.pointerId);
  });

  canvasViewport.addEventListener('pointermove', (e) => {
    if (!state.isPanning) return;
    state.panX = e.clientX - state.startX;
    state.panY = e.clientY - state.startY;
    updateTransform();
  });

  const endPan = (e) => {
    if (state.isPanning) {
      state.isPanning = false;
      canvasViewport.classList.remove('panning');
      try { canvasViewport.releasePointerCapture(e.pointerId); } catch(_) {}
    }
  };

  canvasViewport.addEventListener('pointerup', endPan);
  canvasViewport.addEventListener('pointercancel', endPan);

  canvasViewport.addEventListener('dragover', (e) => {
    e.preventDefault();
    emptyDropzone.classList.add('drag-over');
  });

  canvasViewport.addEventListener('dragleave', (e) => {
    e.preventDefault();
    emptyDropzone.classList.remove('drag-over');
  });

  canvasViewport.addEventListener('drop', (e) => {
    e.preventDefault();
    emptyDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (e.dataTransfer.files.length > 1) {
        // Switch to batch mode automatically
        switchMode('batch');
        handleBatchFilesSelected(e.dataTransfer.files);
      } else {
        handleImageFile(e.dataTransfer.files[0]);
      }
    }
  });

  btnSelectImage.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
    }
  });

  btnZoomIn.addEventListener('click', () => {
    state.zoom = Math.min(state.zoom * 1.2, 25.0);
    updateTransform();
  });

  btnZoomOut.addEventListener('click', () => {
    state.zoom = Math.max(state.zoom / 1.2, 0.05);
    updateTransform();
  });

  btnFit.addEventListener('click', fitToViewport);
  btnActualSize.addEventListener('click', setActualSize);
  btnRotate.addEventListener('click', () => {
    state.rotation = (state.rotation + 90) % 360;
    updateTransform();
  });
  btnReset.addEventListener('click', () => {
    state.rotation = 0;
    fitToViewport();
  });

  tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId)?.classList.add('active');
    });
  });

  // ── 4. Single File SSE Stream Execution ──
  btnRunOcr.addEventListener('click', async () => {
    if (!state.currentFile) {
      showToast('Lütfen önce bir görsel yükleyin!', 'error');
      return;
    }
    if (state.isProcessing) return;

    state.isProcessing = true;
    btnRunOcr.disabled = true;
    btnRunText.textContent = '⚡ Canlı Akıyor...';
    clearBoundingBoxes();

    const selectedModel = modelSelect.value;
    statModel.textContent = selectedModel;
    statLatency.textContent = 'İşleniyor...';

    markdownOutput.innerHTML = `
      <div class="streaming-header-pill">
        <span class="pulse-dot"></span>
        <span>${selectedModel} Canlı Token Akışı (SSE)</span>
      </div>
      <div id="liveStreamText" style="white-space: pre-wrap; font-family: var(--font-sans); line-height: 1.6;"></div>
      <span class="streaming-cursor"></span>
    `;
    const liveStreamText = document.getElementById('liveStreamText');
    plainTextOutput.value = '';

    const formData = new FormData();
    formData.append('file', state.currentFile);
    formData.append('model', selectedModel);

    let accumulatedText = '';
    let finalResultData = null;

    try {
      const response = await fetch('/api/ocr/stream', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Sunucu hatası' }));
        throw new Error(errData.detail || 'OCR akışı başlatılamadı.');
      }

      const reader = response.body.getReader();
      state.activeStreamReader = reader;
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonPayload = trimmed.replace(/^data:\s*/, '');
          let data;
          try {
            data = JSON.parse(jsonPayload);
          } catch (pe) {
            continue;
          }

          if (data.event === 'token') {
            accumulatedText = data.accumulated || (accumulatedText + (data.token || ''));
            if (liveStreamText) liveStreamText.textContent = accumulatedText;
            plainTextOutput.value = accumulatedText;
            statChars.textContent = accumulatedText.length;
            statWords.textContent = accumulatedText ? accumulatedText.trim().split(/\s+/).length : 0;
          } else if (data.event === 'done') {
            finalResultData = data;
          } else if (data.event === 'error') {
            throw new Error(data.error || 'Model çıkarım hatası');
          }
        }
      }

      if (finalResultData) {
        state.lastResult = finalResultData;
        displayResults(finalResultData);
        const warmBadge = finalResultData.is_warm ? ' (Sıcak Önbellek ⚡)' : '';
        showToast(`OCR Tamamlandı! (${finalResultData.elapsed_sec?.toFixed(2)} sn)${warmBadge}`);
      } else if (accumulatedText) {
        const fallbackRes = {
          status: 'success',
          model_key: selectedModel,
          text: accumulatedText,
          markdown: `### ${selectedModel} Çıktısı\n\n${accumulatedText}`,
          boxes: [],
          device: (hardwareText ? hardwareText.textContent.split(' • ')[0] : 'CPU'),
          elapsed_sec: 0.5
        };
        state.lastResult = fallbackRes;
        displayResults(fallbackRes);
        showToast('OCR Tamamlandı!');
      }

    } catch (err) {
      console.error('OCR Error:', err);
      markdownOutput.innerHTML = `<div style="color: #ef4444; padding: 20px;"><strong>Hata:</strong> ${err.message}</div>`;
      showToast(err.message, 'error');
    } finally {
      state.isProcessing = false;
      state.activeStreamReader = null;
      btnRunOcr.disabled = false;
      btnRunText.textContent = 'OCR Çalıştır';
    }
  });

  function displayResults(result) {
    const rawText = result.text || '';
    const markdownText = result.markdown || rawText;
    const elapsedSec = result.elapsed_sec || 0;
    const device = result.device || 'Apple Silicon (MPS)';
    const modelKey = result.model_key || modelSelect.value;
    const boxes = result.boxes || [];

    const parseMarkdown = (typeof marked !== 'undefined' && typeof marked.parse === 'function') 
      ? marked.parse 
      : (typeof marked === 'function' ? marked : null);

    if (parseMarkdown) {
      markdownOutput.innerHTML = parseMarkdown(markdownText);
    } else {
      markdownOutput.textContent = markdownText;
    }

    plainTextOutput.value = rawText;
    jsonOutput.textContent = JSON.stringify(result, null, 2);

    statModel.textContent = modelKey;
    statLatency.textContent = `${elapsedSec.toFixed(3)} sn`;
    statDevice.textContent = device;
    statWords.textContent = rawText ? rawText.trim().split(/\s+/).length : 0;
    statChars.textContent = rawText.length;
    statBoxes.textContent = `${boxes.length} adet`;

    if (hardwareText) {
      hardwareText.textContent = `${device} • ${elapsedSec.toFixed(2)}s`;
    }

    renderBoundingBoxes(boxes);
  }

  function renderBoundingBoxes(boxes) {
    clearBoundingBoxes();
    if (!boxes || !boxes.length || !state.imageWidth) return;

    boundingBoxSvg.setAttribute('viewBox', `0 0 ${state.imageWidth} ${state.imageHeight}`);
    boundingBoxSvg.style.width = `${state.imageWidth}px`;
    boundingBoxSvg.style.height = `${state.imageHeight}px`;

    boxes.forEach((item) => {
      const box = item.box || [];
      const text = item.text || '';
      if (box.length >= 4) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', box[0]);
        rect.setAttribute('y', box[1]);
        rect.setAttribute('width', box[2] - box[0]);
        rect.setAttribute('height', box[3] - box[1]);
        rect.setAttribute('class', 'ocr-box');

        if (text) {
          const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
          title.textContent = `Metin: ${text}`;
          rect.appendChild(title);
        }
        boundingBoxSvg.appendChild(rect);
      }
    });
  }

  function clearBoundingBoxes() {
    boundingBoxSvg.innerHTML = '';
  }

  btnCopyText.addEventListener('click', () => {
    const text = plainTextOutput.value;
    if (!text || !text.trim()) {
      showToast('Kopyalanacak metin bulunamadı!', 'error');
      return;
    }
    navigator.clipboard.writeText(text).then(() => {
      showToast('Metin panoya kopyalandı! 📋');
    }).catch(() => {
      plainTextOutput.select();
      document.execCommand('copy');
      showToast('Metin panoya kopyalandı! 📋');
    });
  });

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`İndirildi: ${filename}`);
  }

  btnExportTxt.addEventListener('click', () => {
    const text = plainTextOutput.value;
    if (!text || !text.trim()) {
      showToast('İndirilecek metin bulunamadı!', 'error');
      return;
    }
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    downloadBlob(blob, 'ocr_cikti.txt');
  });

  btnExportJson.addEventListener('click', () => {
    if (!state.lastResult) {
      showToast('İndirilecek OCR sonucu bulunamadı!', 'error');
      return;
    }
    const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], { type: 'application/json;charset=utf-8' });
    downloadBlob(blob, 'ocr_cikti.json');
  });

  btnExportMd.addEventListener('click', () => {
    const content = state.lastResult?.markdown || plainTextOutput.value;
    if (!content || !content.trim()) {
      showToast('İndirilecek Markdown içeriği bulunamadı!', 'error');
      return;
    }
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    downloadBlob(blob, 'ocr_cikti.md');
  });


  // ── 5. Batch Mode Logic & Multi-File Handlers ──
  const validExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'];

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function handleBatchFilesSelected(fileList) {
    const images = [];
    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      const ext = '.' + f.name.split('.').pop().toLowerCase();
      if (validExtensions.includes(ext) || f.type.startsWith('image/')) {
        images.push(f);
      }
    }

    if (images.length === 0) {
      showToast('Seçilen dosyalarda desteklenen formatta görsel bulunamadı.', 'error');
      return;
    }

    state.batchFiles = images;
    batchFileCountText.textContent = `${images.length} dosya seçildi`;
    btnRunBatch.disabled = false;
    renderFileChips(images);
    showToast(`${images.length} adet fiş/görsel toplu tarama için hazır!`);
  }

  function renderFileChips(files) {
    batchFileChips.innerHTML = '';
    const maxShow = 100;
    const slice = files.slice(0, maxShow);

    slice.forEach((f) => {
      const chip = document.createElement('div');
      chip.className = 'file-chip-item';
      chip.innerHTML = `
        <span class="file-chip-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>
        <span class="file-chip-size">${formatBytes(f.size)}</span>
      `;
      batchFileChips.appendChild(chip);
    });

    if (files.length > maxShow) {
      const more = document.createElement('div');
      more.className = 'empty-chips-hint';
      more.textContent = `... ve ${files.length - maxShow} dosya daha`;
      batchFileChips.appendChild(more);
    }
  }

  btnSelectBatchFolder.addEventListener('click', () => batchFolderInput.click());
  btnSelectBatchFiles.addEventListener('click', () => batchFilesInput.click());

  batchFolderInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleBatchFilesSelected(e.target.files);
    }
  });

  batchFilesInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleBatchFilesSelected(e.target.files);
    }
  });

  batchDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    batchDropzone.classList.add('drag-over');
  });

  batchDropzone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    batchDropzone.classList.remove('drag-over');
  });

  batchDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    batchDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleBatchFilesSelected(e.dataTransfer.files);
    }
  });

  // ── 6. Live SSE Streaming Batch Runner ──
  btnRunBatch.addEventListener('click', async () => {
    if (!state.batchFiles || state.batchFiles.length === 0) {
      showToast('Lütfen önce taranacak fişleri seçin!', 'error');
      return;
    }
    if (state.isBatchProcessing) return;

    state.isBatchProcessing = true;
    state.batchResults = [];
    state.batchStartTime = Date.now();
    btnRunBatch.disabled = true;
    btnRunBatchText.textContent = '⚡ Toplu Taranıyor...';

    // Reset UI indicators
    batchExportToolbar.style.display = 'none';
    batchActiveItemCard.style.display = 'flex';
    batchTableBody.innerHTML = '';
    batchProgressFill.style.width = '0%';
    batchProgressCounter.textContent = `%0 — 0 / ${state.batchFiles.length} Fiş`;
    batchProgressHeading.textContent = 'Toplu Tarama Başlatılıyor...';
    batchStatTotal.textContent = state.batchFiles.length;
    batchStatSuccess.textContent = '0';
    batchStatFailed.textContent = '0';
    batchStatTime.textContent = '0.00s';

    const selectedModel = modelSelect.value;
    const selectedFormat = batchFormatSelect.value;

    const formData = new FormData();
    state.batchFiles.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('model', selectedModel);
    formData.append('format', selectedFormat);

    let successCount = 0;
    let failedCount = 0;
    let finalBatchSummary = null;

    try {
      const response = await fetch('/api/batch/stream', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Sunucu hatası' }));
        throw new Error(errData.detail || 'Toplu işlem başlatılamadı.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonPayload = trimmed.replace(/^data:\s*/, '');
          let data;
          try {
            data = JSON.parse(jsonPayload);
          } catch (pe) {
            continue;
          }

          if (data.event === 'start') {
            batchProgressHeading.textContent = `${selectedModel} Sıcak Önbellek Taraması`;
          } else if (data.event === 'progress') {
            const idx = data.index;
            const total = data.total;
            const pct = data.percent;
            const isOk = data.status === 'success';

            if (isOk) successCount++;
            else failedCount++;

            // Update Progress Bar
            batchProgressFill.style.width = `${pct}%`;
            batchProgressCounter.textContent = `%${pct} — ${idx} / ${total} Fiş`;

            // Update Metrics Cards
            batchStatSuccess.textContent = successCount;
            batchStatFailed.textContent = failedCount;
            const elapsedSinceStart = ((Date.now() - state.batchStartTime) / 1000).toFixed(2);
            const avgSpeed = (elapsedSinceStart / idx).toFixed(2);
            batchStatTime.textContent = `${elapsedSinceStart}s (${avgSpeed}s/fiş)`;

            // Update Active Preview Card
            batchActiveFilename.textContent = data.filename;
            batchActiveLatency.textContent = `${data.elapsed_sec?.toFixed(3)}s`;
            batchActivePreview.textContent = isOk 
              ? (data.preview || 'Metin başarıyla tanındı.')
              : `HATA: ${data.error || 'İşlenemedi'}`;

            // Append row to results table
            appendBatchTableRow(idx, data);
            state.batchResults.push(data);

          } else if (data.event === 'done') {
            finalBatchSummary = data;
          } else if (data.event === 'error') {
            throw new Error(data.error || 'Toplu işlem sırasında hata oluştu.');
          }
        }
      }

      // Finalize
      batchProgressFill.style.width = '100%';
      batchProgressHeading.textContent = 'Toplu Tarama Tamamlandı!';
      batchActiveItemCard.style.display = 'none';

      if (finalBatchSummary) {
        state.lastBatchData = finalBatchSummary;
        batchExportToolbar.style.display = 'flex';
        batchSummaryText.textContent = `Toplam ${state.batchFiles.length} fiş işlendi (${successCount} başarılı, ${failedCount} hata). Çıktılar indirilebilir.`;
        showToast(`Toplu işlem bitti! (${successCount} başarılı / ${state.batchFiles.length} fiş)`);
      }

    } catch (err) {
      console.error('Batch Error:', err);
      batchProgressHeading.textContent = 'Toplu İşlem Hatası!';
      showToast(err.message, 'error');
    } finally {
      state.isBatchProcessing = false;
      btnRunBatch.disabled = false;
      btnRunBatchText.textContent = '⚡ Toplu Taramayı Başlat';
    }
  });

  function appendBatchTableRow(index, item) {
    const isOk = item.status === 'success';
    const statusPill = isOk
      ? `<span class="status-pill status-pill-success">✓ Başarılı</span>`
      : `<span class="status-pill status-pill-error">✗ Hata</span>`;

    const row = document.createElement('tr');
    row.setAttribute('data-filename', (item.filename || '').toLowerCase());
    row.setAttribute('data-text', (item.preview || item.error || '').toLowerCase());

    const safeFilename = escapeHtml(item.filename || '');
    const safePreview = isOk 
      ? (escapeHtml(item.preview) || '-') 
      : `<span style="color: #ef4444;">${escapeHtml(item.error || 'İşleme hatası')}</span>`;

    row.innerHTML = `
      <td style="font-family: var(--font-mono); color: var(--text-muted);">${index}</td>
      <td style="font-weight: 600; font-family: var(--font-mono); color: var(--text-primary);">${safeFilename}</td>
      <td>${statusPill}</td>
      <td style="font-family: var(--font-mono); color: var(--accent-blue);">${item.elapsed_sec?.toFixed(3)}s</td>
      <td style="font-family: var(--font-mono);">${item.char_count || 0}</td>
      <td style="font-family: var(--font-mono);">${item.word_count || 0}</td>
      <td style="font-family: var(--font-mono); max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        ${safePreview}
      </td>
    `;
    batchTableBody.appendChild(row);
  }

  // ── 7. Batch Export & Download Handlers ──
  function triggerFileDownload(url, defaultName) {
    const a = document.createElement('a');
    a.href = url;
    if (defaultName) a.setAttribute('download', defaultName);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  btnBatchDownloadZip.addEventListener('click', () => {
    if (!state.lastBatchData?.zip_url) {
      showToast('İndirilecek ZIP arşivi bulunamadı.', 'error');
      return;
    }
    triggerFileDownload(state.lastBatchData.zip_url);
    showToast('Tüm çıktılar (.ZIP) indiriliyor...');
  });

  btnBatchDownloadCsv.addEventListener('click', () => {
    if (!state.lastBatchData?.csv_url) {
      showToast('İndirilecek summary.csv bulunamadı.', 'error');
      return;
    }
    triggerFileDownload(state.lastBatchData.csv_url);
    showToast('summary.csv (Excel) indiriliyor...');
  });

  btnBatchDownloadJson.addEventListener('click', () => {
    if (!state.lastBatchData?.json_url) {
      showToast('İndirilecek summary.json bulunamadı.', 'error');
      return;
    }
    triggerFileDownload(state.lastBatchData.json_url);
    showToast('summary.json indiriliyor...');
  });

  // Table Search Filter
  batchTableSearch.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    const rows = batchTableBody.querySelectorAll('tr:not(.empty-row)');
    rows.forEach((r) => {
      const fname = r.getAttribute('data-filename') || '';
      const text = r.getAttribute('data-text') || '';
      if (!q || fname.includes(q) || text.includes(q)) {
        r.style.display = '';
      } else {
        r.style.display = 'none';
      }
    });
  });

  // ── 8. Global Keyboard Shortcuts ──
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'o') {
      e.preventDefault();
      if (state.mode === 'single') fileInput.click();
      else batchFilesInput.click();
    } else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (state.mode === 'single') btnRunOcr.click();
      else btnRunBatch.click();
    } else if ((e.metaKey || e.ctrlKey) && e.key === '0') {
      e.preventDefault();
      fitToViewport();
    } else if ((e.metaKey || e.ctrlKey) && e.key === '1') {
      e.preventDefault();
      setActualSize();
    }
  });

  // ── 9. Initial Backend Info Fetch ──
  fetch('/api/models')
    .then(r => r.json())
    .then(data => {
      if (data.device && hardwareText) {
        hardwareText.textContent = data.device;
      }
    })
    .catch(() => {});
});
