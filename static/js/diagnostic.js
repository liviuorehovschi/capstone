document.getElementById('uploadForm').onsubmit = async function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    const loadingContainer = document.getElementById('loadingContainer');
    const resultDiv = document.getElementById('result');
    const mapButtonRow = document.getElementById('mapButtonRow');

    // Show loading animation
    loadingContainer.style.display = 'block';
    resultDiv.style.display = 'none';
    mapButtonRow.style.display = 'none'; // Hide map buttons on new submit

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Received result:', result);

        if (result.error) {
            resultDiv.innerHTML = `<p class="error">Error: ${result.error}</p>`;
        } else {
            const diagnosis = formatDiagnosisMessage(result.class);
            const confidenceText = getConfidenceInterpretation(result.confidence);

            resultDiv.innerHTML = `
                <div class="result-card" style="display: block;">
                    <h3 class="result-header ${diagnosis.class}">${diagnosis.message}</h3>
                    <div class="confidence-container">
                        <span>Confidence: ${(result.confidence * 100).toFixed(2)}%</span>
                        <div class="tooltip">
                            <span class="help-icon">ⓘ</span>
                            <span class="tooltip-text">
                                <strong>Understanding Confidence:</strong><br>
                                The confidence score indicates the model's certainty in its prediction.<br><br>
                                ${confidenceText}
                            </span>
                        </div>
                    </div>
                </div>`;

            // ✅ Show the Grad-CAM and Saliency buttons
            mapButtonRow.style.display = 'flex';
            document.getElementById('showGradcamBtn').style.display = 'inline-block';
            document.getElementById('showSaliencyBtn').style.display = 'inline-block';
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
    } finally {
        loadingContainer.style.display = 'none';
        resultDiv.style.display = 'block';
    }
};

// Grad-CAM button logic
document.getElementById('showGradcamBtn').addEventListener('click', async function () {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/gradcam_image', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const gradcamPreview = document.getElementById('gradcamPreview');
            gradcamPreview.src = url;
            gradcamPreview.style.display = 'block';

            // 🔥 Hide Grad-CAM button after click
            document.getElementById('showGradcamBtn').style.display = 'none';
        } else {
            console.error("Grad-CAM fetch error:", await response.text());
        }
    } catch (err) {
        console.error("Grad-CAM error:", err);
    }
});

// Saliency Map button logic
document.getElementById('showSaliencyBtn').addEventListener('click', async function () {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/saliency_image', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const saliencyPreview = document.getElementById('saliencyPreview');
            saliencyPreview.src = url;
            saliencyPreview.style.display = 'block';

            // 🔥 Hide Saliency button after click
            document.getElementById('showSaliencyBtn').style.display = 'none';
        } else {
            console.error("Saliency fetch error:", await response.text());
        }
    } catch (err) {
        console.error("Saliency error:", err);
    }
});

// Reset previews + buttons when a new image is selected
document.getElementById('fileInput').addEventListener('change', function (e) {
    const preview = document.getElementById('preview');
    const previewContainer = document.getElementById('imagePreview');
    const saliencyPreview = document.getElementById('saliencyPreview');
    const gradcamPreview = document.getElementById('gradcamPreview');
    const mapButtonRow = document.getElementById('mapButtonRow');

    if (e.target.files && e.target.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';

            // Reset map previews and buttons
            saliencyPreview.style.display = 'none';
            saliencyPreview.src = '';
            gradcamPreview.style.display = 'none';
            gradcamPreview.src = '';
            mapButtonRow.style.display = 'none';
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});

// Utilities
function getConfidenceInterpretation(confidence) {
    const percentage = (confidence * 100).toFixed(2);
    if (percentage >= 95) {
        return `Very high confidence (${percentage}%). The model shows a strong certainty in this diagnosis, suggesting a highly reliable prediction within the model's capabilities.`;
    } else if (percentage >= 85) {
        return `High confidence (${percentage}%). The model shows good certainty in this diagnosis, though further clinical verification is recommended.`;
    } else if (percentage >= 70) {
        return `Moderate confidence (${percentage}%). While the model leans towards this diagnosis, additional testing and clinical correlation is strongly advised.`;
    } else {
        return `Lower confidence (${percentage}%). This prediction should be treated as preliminary, requiring thorough clinical verification and additional testing.`;
    }
}

function formatDiagnosisMessage(className) {
    switch (className) {
        case 'lung_n':
            return { message: "The tissue appears normal", class: "diagnosis-normal" };
        case 'lung_aca':
            return { message: "Signs of adenocarcinoma detected", class: "diagnosis-cancer" };
        case 'lung_scc':
            return { message: "Signs of squamous cell carcinoma detected", class: "diagnosis-cancer" };
        default:
            return { message: "Unknown tissue type", class: "" };
    }
}
