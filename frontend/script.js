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
    const resetBtn = document.getElementById('reset-btn');

    let selectedFile = null;

    // --- Upload Logic ---
    
    // Click to upload
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and drop
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

    // File input change
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

    // Remove image
    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    // --- API Interaction ---

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading state
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        // Create FormData
        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            // Replace with actual API endpoint
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
            uploadSection.classList.remove('hidden');
        }
    });

    function showResult(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        // Set Images
        resultOriginalImg.src = imagePreview.src;
        if (data.gradcam_url) {
            resultGradcamImg.src = data.gradcam_url;
            resultGradcamImg.parentElement.classList.remove('hidden');
        } else {
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

        // Mostrar clase mayoritaria
        resultMajorityClass.textContent = `Diagnóstico: ${predictedClass}`;

        // Update colors and text based on result
        if (isCancer) {
            resultTitle.textContent = "ALTO RIESGO";
            resultTitle.className = "result-title-danger";
            resultMajorityClass.style.color = "var(--danger)";
            confidenceCircle.className.baseVal = "circle danger";
            
            resultDetails.innerHTML = `
                <h3 style="color: var(--danger)">Posible Melanoma Detectado</h3>
                <p style="color: var(--text-muted); font-size: 0.9rem;">
                    El modelo indica características asociadas a malignidad. Se recomienda fuertemente consultar con un dermatólogo para un diagnóstico profesional.
                </p>
            `;
        } else {
            resultTitle.textContent = "BAJO RIESGO";
            resultTitle.className = "result-title-success";
            resultMajorityClass.style.color = "var(--success)";
            confidenceCircle.className.baseVal = "circle success";
            
            resultDetails.innerHTML = `
                <h3 style="color: var(--success)">Sin signos evidentes de malignidad</h3>
                <p style="color: var(--text-muted); font-size: 0.9rem;">
                    El modelo no detectó características fuertemente asociadas a melanoma. Sin embargo, mantén revisiones periódicas con tu médico.
                </p>
            `;
        }

        // Render probabilities list
        if (data.probabilities) {
            let probsHtml = '<h3>Probabilidades por Clase</h3>';
            for (const [className, prob] of Object.entries(data.probabilities)) {
                const probPercent = (prob * 100).toFixed(1);
                // Assign color based on class
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
    }

    // Reset Flow
    resetBtn.addEventListener('click', () => {
        // Reset state
        selectedFile = null;
        fileInput.value = '';
        
        // Reset UI
        resultSection.classList.add('hidden');
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        uploadSection.classList.remove('hidden');
        
        // Reset animation
        confidenceCircle.setAttribute('stroke-dasharray', '0, 100');
    });
});
