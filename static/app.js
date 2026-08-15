/**
 * DocuMind UI Frontend Script
 * Interacts with FastAPI Backend to handle PDF uploads and queries.
 */

// Application State
const state = {
    isConnected: false,
    isIndexed: false,
    activeDocument: null,
    // Store citations from the latest query response, keyed by chunk_id
    citationsMap: new Map(),
    isUploading: false,
    isQuerying: false
};

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const statusFilename = document.getElementById('status-filename');
const statusPercent = document.getElementById('status-percent');
const progressFill = document.getElementById('progress-fill');
const activeDocContainer = document.getElementById('active-document');
const apiStatusText = document.getElementById('api-status');
const pulseDot = document.querySelector('.pulse-dot');

const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const chatSubStatus = document.getElementById('chat-sub-status');

const citationInspector = document.getElementById('citation-inspector');
const inspectorContent = document.getElementById('inspector-content');
const closeInspectorBtn = document.getElementById('close-inspector-btn');

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    checkBackendConnection();
    // Poll connection status every 10 seconds
    setInterval(checkBackendConnection, 10000);
    setupUploadHandlers();
    setupChatHandlers();
    setupInspectorHandlers();
});

// 1. Connection Monitoring
async function checkBackendConnection() {
    try {
        const response = await fetch('/');
        if (response.ok) {
            setConnectionStatus(true, 'System Active');
            
            // Check if there is an active index ready on startup
            try {
                // Fetch the service metadata using the root endpoint or options
                const metaResponse = await fetch('/static/index.html'); // just to check server responsiveness
                // Let's do a simple call or load index state
            } catch (err) {}
        } else {
            setConnectionStatus(false, 'API Error');
        }
    } catch (error) {
        setConnectionStatus(false, 'Backend Offline');
    }
}

function setConnectionStatus(connected, message) {
    state.isConnected = connected;
    apiStatusText.textContent = message;
    
    if (connected) {
        pulseDot.className = 'pulse-dot online';
    } else {
        pulseDot.className = 'pulse-dot offline';
        showToast('Connection to backend lost.', 'error');
        disableInterface();
    }
}

// 2. Drag & Drop File Upload
function setupUploadHandlers() {
    // Click on drop zone triggers file input
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag-over styling classes
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

function handleFileUpload(file) {
    if (!state.isConnected) {
        showToast('Cannot upload: Backend is offline.', 'error');
        return;
    }
    if (state.isUploading || state.isQuerying) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
        showToast('Only PDF files are supported.', 'error');
        return;
    }

    state.isUploading = true;
    uploadStatus.hidden = false;
    statusFilename.textContent = file.name;
    updateUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    // Using XMLHttpRequest to support upload progress indicators
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percentComplete = Math.round((e.loaded / e.total) * 100);
            // Limit progress to 90% until backend indexing completes
            updateUploadProgress(Math.min(percentComplete, 90));
        }
    };

    xhr.onload = () => {
        state.isUploading = false;
        if (xhr.status === 200) {
            updateUploadProgress(100);
            const response = JSON.parse(xhr.responseText);
            showToast('Document uploaded and indexed successfully!', 'success');
            
            // Set document state
            state.isIndexed = true;
            state.activeDocument = {
                name: file.name,
                size: formatBytes(file.size),
                chunks: response.chunks_count
            };
            
            renderActiveDocument();
            enableInterface();
            
            // Post Success Message to Chat
            appendMessage('system', 'System Assistant', 
                `Success! The document <strong>${file.name}</strong> (${formatBytes(file.size)}) has been parsed page-by-page, cleaned, and split into <strong>${response.chunks_count} semantic chunks</strong>. Parallel FAISS and BM25 indices have been successfully built. You can now ask questions!`
            );
            
            // Reset upload box
            setTimeout(() => {
                uploadStatus.hidden = true;
                updateUploadProgress(0);
            }, 3000);
        } else {
            let errorMsg = 'Upload failed.';
            try {
                const err = JSON.parse(xhr.responseText);
                errorMsg = err.detail || errorMsg;
            } catch (e) {}
            showToast(errorMsg, 'error');
            resetUploadArea();
        }
    };

    xhr.onerror = () => {
        state.isUploading = false;
        showToast('Network error during file upload.', 'error');
        resetUploadArea();
    };

    xhr.send(formData);
}

function updateUploadProgress(percent) {
    progressFill.style.width = `${percent}%`;
    statusPercent.textContent = `${percent}%`;
}

function resetUploadArea() {
    uploadStatus.hidden = true;
    updateUploadProgress(0);
    state.isUploading = false;
}

function renderActiveDocument() {
    if (state.activeDocument) {
        activeDocContainer.innerHTML = `
            <div class="active-doc-card">
                <i class="fa-solid fa-file-pdf doc-file-icon"></i>
                <div class="doc-details">
                    <span class="doc-name" title="${state.activeDocument.name}">${state.activeDocument.name}</span>
                    <span class="doc-meta">${state.activeDocument.size} • ${state.activeDocument.chunks} chunks</span>
                </div>
            </div>
        `;
        chatSubStatus.textContent = `Asking questions grounded in: ${state.activeDocument.name}`;
    } else {
        activeDocContainer.innerHTML = `
            <div class="empty-doc-state">
                <i class="fa-solid fa-circle-exclamation"></i>
                <p>No document indexed yet</p>
            </div>
        `;
        chatSubStatus.textContent = 'Upload a document to start asking questions';
    }
}

// 3. Chat Session Interactivity
function setupChatHandlers() {
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (query) {
            handleQuerySubmit(query);
        }
    });

    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = '';
        state.citationsMap.clear();
        closeInspector();
        appendMessage('system', 'DocuMind System Assistant', 'Chat session cleared. Upload a document or ask questions to begin.');
        showToast('Chat history cleared.', 'info');
    });

    // Event delegation for clickable citations in chat bubbles
    chatMessages.addEventListener('click', (e) => {
        if (e.target.classList.contains('citation-tag')) {
            const chunkId = e.target.getAttribute('data-chunk-id');
            openCitationInspector(chunkId);
        }
    });
}

async function handleQuerySubmit(query) {
    if (!state.isIndexed || state.isQuerying) return;
    
    state.isQuerying = true;
    queryInput.value = '';
    
    // Disable inputs
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Append User Message bubble
    appendMessage('user', 'You', query);

    // Append Thinking Skeleton bubble
    const thinkingBubbleId = appendThinkingBubble();

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: 4
            })
        });

        // Remove thinking bubble
        removeBubble(thinkingBubbleId);

        if (response.ok) {
            const data = await response.json();
            
            // Map citations so they can be looked up instantly by inspectCitation
            data.citations.forEach(citation => {
                state.citationsMap.set(citation.chunk_id, citation);
            });

            // Parse response to make [chunk_xxxx] tags clickable html
            const formattedAnswer = formatAnswerCitations(data.answer);

            // Append assistant response bubble
            appendMessage('assistant', 'DocuMind Agent', formattedAnswer);
        } else {
            const err = await response.json();
            appendMessage('system', 'System Error', err.detail || 'Failed to generate answer.');
            showToast('RAG pipeline query failed.', 'error');
        }
    } catch (error) {
        removeBubble(thinkingBubbleId);
        appendMessage('system', 'System Error', 'Network error. Could not reach server.');
        showToast('Query request dropped.', 'error');
    } finally {
        state.isQuerying = false;
        enableInterface();
    }
}

function appendMessage(sender, senderName, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const iconClass = sender === 'user' ? 'fa-user' : (sender === 'system' ? 'fa-gears' : 'fa-robot');
    const headerHtml = `<div class="message-header"><i class="fa-solid ${iconClass}"></i> <span>${senderName}</span></div>`;

    messageDiv.innerHTML = `
        <div class="message-content">
            ${headerHtml}
            <p>${text}</p>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendThinkingBubble() {
    const id = 'thinking-' + Date.now();
    const bubble = document.createElement('div');
    bubble.className = 'message assistant';
    bubble.id = id;
    
    bubble.innerHTML = `
        <div class="message-content">
            <div class="message-header"><i class="fa-solid fa-robot"></i> <span>DocuMind Agent</span></div>
            <div class="thinking-container">
                <div class="skeleton-line"></div>
                <div class="skeleton-line medium"></div>
                <div class="skeleton-line short"></div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeBubble(id) {
    const bubble = document.getElementById(id);
    if (bubble) {
        bubble.remove();
    }
}

function formatAnswerCitations(answerText) {
    // Regex matches [chunk_xxxx] tags
    const citationRegex = /\[(chunk_\d+)\]/g;
    return answerText.replace(citationRegex, (match, chunkId) => {
        return `<span class="citation-tag" data-chunk-id="${chunkId}">${match}</span>`;
    });
}

// 4. Citation Inspector panel logic
function setupInspectorHandlers() {
    closeInspectorBtn.addEventListener('click', closeInspector);
}

function openCitationInspector(chunkId) {
    const citation = state.citationsMap.get(chunkId);
    
    if (!citation) {
        showToast(`Metadata for ${chunkId} not cached.`, 'warning');
        return;
    }

    // Populate inspector
    inspectorContent.innerHTML = `
        <div class="inspector-meta-grid">
            <div class="meta-item">
                <div class="meta-label">Chunk Reference</div>
                <div class="meta-val" style="color: var(--accent-indigo);">${citation.chunk_id}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Source Document</div>
                <div class="meta-val" title="${citation.source_doc}">${citation.source_doc}</div>
            </div>
            <div class="meta-item" style="grid-column: span 2;">
                <div class="meta-label">Page Number</div>
                <div class="meta-val"><i class="fa-solid fa-file-lines"></i> Page ${citation.page_number}</div>
            </div>
        </div>
        
        <div class="inspector-text-block">
            <strong><i class="fa-solid fa-align-left"></i> Chunk Text Content</strong>
            <p>${escapeHTML(citation.text)}</p>
        </div>
    `;

    // Slide open panel
    citationInspector.classList.remove('closed');
}

function closeInspector() {
    citationInspector.classList.add('closed');
}

// Interface State Toggles
function disableInterface() {
    queryInput.disabled = true;
    sendBtn.disabled = true;
}

function enableInterface() {
    if (state.isConnected && state.isIndexed && !state.isQuerying) {
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.placeholder = 'Ask a question about the document...';
    } else if (state.isConnected && !state.isIndexed) {
        queryInput.disabled = true;
        sendBtn.disabled = true;
        queryInput.placeholder = 'Upload a PDF to start...';
    }
}

// Toast Alerts
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'fa-circle-info';
    if (type === 'success') icon = 'fa-circle-check';
    if (type === 'error') icon = 'fa-triangle-exclamation';
    if (type === 'warning') icon = 'fa-circle-exclamation';

    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Auto remove after 4.5 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}

// Helper Utilities
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function escapeHTML(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
