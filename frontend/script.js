document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    
    const resultTitle = document.getElementById('result-title');
    const resultMajorityClass = document.getElementById('result-majority-class');
    const resultOriginalImg = document.getElementById('result-original-img');
    const resultGradcamImg = document.getElementById('result-gradcam-img');
    
    const confidenceCircle = document.getElementById('confidence-circle');
    const confidenceText = document.getElementById('confidence-text');
    const resultDetails = document.getElementById('result-details');
    const probabilitiesList = document.getElementById('probabilities-list');
    
    // Elementos de las nuevas funcionalidades
    const resetBtn = document.getElementById('reset-btn');
    const pdfBtn = document.getElementById('pdf-btn');
    const historySidebar = document.getElementById('history-sidebar');
    const toggleHistoryBtn = document.getElementById('toggle-history-btn');
    const historyList = document.getElementById('history-list');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    const modelSelector = document.getElementById('model-selector');
    const tabularSection = document.getElementById('tabular-section');
    
    let selectedFile = null;

    // --- UI Logic ---
    modelSelector.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'resnet50') {
            tabularSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        } else if (val === 'tabular') {
            tabularSection.classList.remove('hidden');
            uploadSection.classList.add('hidden');
        } else if (val === 'hibrido') {
            tabularSection.classList.remove('hidden');
            uploadSection.classList.remove('hidden');
        }
    });

    // --- Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Por favor, selecciona un archivo de imagen válido.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    // --- API Interaction ---
    analyzeBtn.addEventListener('click', async () => {
        const modelType = modelSelector.value;
        
        if (modelType !== 'tabular' && !selectedFile) {
            alert('Por favor, selecciona una imagen para el modelo.');
            return;
        }

        // Show loading state
        uploadSection.classList.add('hidden');
        tabularSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        // Create FormData
        const formData = new FormData();
        formData.append('model_type', modelType);
        
        if (modelType !== 'tabular') {
            formData.append('image', selectedFile);
        }

        if (modelType !== 'resnet50') {
            formData.append('asymmetry', document.getElementById('asymmetry').value);
            formData.append('pigment_network', document.getElementById('pigment_network').value);
            formData.append('dots_globules', document.getElementById('dots_globules').value);
            formData.append('streaks', document.getElementById('streaks').value);
            formData.append('regression_areas', document.getElementById('regression_areas').value);
            formData.append('blue_whitish_veil', document.getElementById('blue_whitish_veil').value);
            
            // Collect checked colors
            const colors = Array.from(document.querySelectorAll('input[name="colors"]:checked'))
                .map(cb => cb.value)
                .join(',');
            formData.append('colors', colors);
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Error en el servidor');
            }

            const data = await response.json();
            showResult(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Hubo un error al procesar la imagen. Verifica que el servidor backend esté corriendo.');
            
            // Reset UI
            loadingSection.classList.add('hidden');
            if (modelType !== 'tabular') uploadSection.classList.remove('hidden');
            if (modelType !== 'resnet50') tabularSection.classList.remove('hidden');
        }
    });

    function showResult(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        // Set Images
        if (data.gradcam_url) {
            resultOriginalImg.src = selectedFile ? imagePreview.src : '';
            resultOriginalImg.parentElement.classList.remove('hidden');
            resultGradcamImg.src = data.gradcam_url;
            resultGradcamImg.parentElement.classList.remove('hidden');
        } else {
            resultOriginalImg.parentElement.classList.add('hidden');
            resultGradcamImg.parentElement.classList.add('hidden');
        }

        const isCancer = data.is_cancer;
        const confidenceValue = (data.confidence * 100).toFixed(1);
        const predictedClass = data.predicted_class || (isCancer ? 'Melanoma' : 'Benigno');
        
        // Setup circle animation
        setTimeout(() => {
            confidenceCircle.setAttribute('stroke-dasharray', `${confidenceValue}, 100`);
        }, 100);

        confidenceText.textContent = `${confidenceValue}%`;
        resultMajorityClass.textContent = `Diagnóstico: ${predictedClass}`;

        // Update colors based on result
        if (isCancer) {
            resultTitle.textContent = "ALTO RIESGO";
            resultTitle.className = "result-title-danger";
            resultMajorityClass.style.color = "var(--danger)";
            confidenceCircle.className.baseVal = "circle danger";
        } else {
            resultTitle.textContent = "BAJO RIESGO";
            resultTitle.className = "result-title-success";
            resultMajorityClass.style.color = "var(--success)";
            confidenceCircle.className.baseVal = "circle success";
        }

        // Lógica de Recomendación Clínica ("El Porqué")
        const colorRecomendacion = isCancer ? "var(--danger)" : "var(--success)";
        const tituloRecomendacion = isCancer ? "Posible Melanoma Detectado" : "Sin signos evidentes de malignidad";
        const recomendacionTexto = data.clinical_recommendation || (isCancer ? 'Se recomienda fuertemente consultar con un dermatólogo para un diagnóstico profesional.' : 'Mantenga revisiones periódicas con su médico.');

        resultDetails.innerHTML = `
            <h3 style="color: ${colorRecomendacion}">${tituloRecomendacion}</h3>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; margin-top: 10px; background: #f8f9fa; padding: 15px; border-left: 4px solid ${colorRecomendacion}; border-radius: 4px;">
                <strong>Contexto Clínico:</strong> ${recomendacionTexto}
            </p>
        `;

        // Render probabilities list
        if (data.probabilities) {
            let probsHtml = '<h3>Probabilidades por Clase</h3>';
            for (const [className, prob] of Object.entries(data.probabilities)) {
                const probPercent = (prob * 100).toFixed(1);
                let barColor = 'var(--primary)';
                if (className.toLowerCase().includes('melanoma')) barColor = 'var(--danger)';
                else if (className.toLowerCase().includes('common nevus')) barColor = 'var(--success)';
                
                probsHtml += `
                    <div class="prob-item">
                        <div class="prob-label">${className}</div>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: ${probPercent}%; background-color: ${barColor}"></div>
                        </div>
                        <div class="prob-value">${probPercent}%</div>
                    </div>
                `;
            }
            probabilitiesList.innerHTML = probsHtml;
            probabilitiesList.classList.remove('hidden');
        } else {
            probabilitiesList.classList.add('hidden');
        }

        // Guardar en el Historial Local
        saveToHistory({
            filename: data.filename || 'Análisis Tabular',
            clase: predictedClass,
            confianza: confidenceValue,
            fecha: new Date().toLocaleString(),
            isCancer: isCancer,
            // Si no hay foto (solo tabular), usamos un placeholder genérico
            imgSrc: (selectedFile && imagePreview.src) ? imagePreview.src : 'https://via.placeholder.com/50/e2e8f0/64748b?text=Tab'
        });
    }

    // --- NUEVAS FUNCIONALIDADES (PDF E HISTORIAL) ---

    // Descarga de PDF
    if (pdfBtn) {
        pdfBtn.addEventListener('click', () => {
            const resultElement = document.querySelector('.result-card');
            const opt = {
                margin:       0.5,
                filename:     'Reporte_Dermatologico_IA.pdf',
                image:        { type: 'jpeg', quality: 0.98 },
                html2canvas:  { scale: 2, useCORS: true }, 
                jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
            };
            html2pdf().set(opt).from(resultElement).save();
        });
    }

    // Toggle Panel Lateral
    if (toggleHistoryBtn && historySidebar) {
        toggleHistoryBtn.addEventListener('click', () => {
            historySidebar.classList.toggle('collapsed');
        });
    }

    function saveToHistory(record) {
        let history = JSON.parse(localStorage.getItem('melanomaHistory')) || [];
        history.unshift(record); 
        // Limitar a los 5 análisis más recientes para ahorrar espacio en memoria
        if (history.length > 5) history.pop(); 
        localStorage.setItem('melanomaHistory', JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        if (!historyList) return;
        let history = JSON.parse(localStorage.getItem('melanomaHistory')) || [];
        historyList.innerHTML = '';
        
        if(history.length === 0) {
            historyList.innerHTML = '<p style="color: #666; font-size: 0.9rem;">No hay análisis recientes.</p>';
            return;
        }

        history.forEach(item => {
            const colorClass = item.isCancer ? 'color: var(--danger);' : 'color: var(--success);';
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <img src="${item.imgSrc}" alt="Miniatura">
                <div class="history-item-info">
                    <strong style="${colorClass}">${item.clase} (${item.confianza}%)</strong>
                    <span>${item.fecha}</span>
                </div>
            `;
            historyList.appendChild(historyItem);
        });
    }

    // Limpiar Historial
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            localStorage.removeItem('melanomaHistory');
            renderHistory();
        });
    }

    // Cargar historial al inicializar la página
    renderHistory();

    // --- Reset Flow ---
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            
            document.querySelectorAll('select').forEach(select => {
                if(select.id !== 'model-selector') select.selectedIndex = 0;
            });
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            
            resultSection.classList.add('hidden');
            previewContainer.classList.add('hidden');
            dropZone.classList.remove('hidden');
            
            modelSelector.dispatchEvent(new Event('change'));
            confidenceCircle.setAttribute('stroke-dasharray', '0, 100');
        });
    }
});