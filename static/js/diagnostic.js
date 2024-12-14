document.getElementById('uploadForm').onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const loadingContainer = document.getElementById('loadingContainer');
    const resultDiv = document.getElementById('result');
    
    // Show loading animation
    loadingContainer.style.display = 'block';
    resultDiv.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Received result:', result); // Debug log

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
        }
    } catch (error) {
        console.error('Error:', error); // Debug log
        resultDiv.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
    } finally {
        // Hide loading animation
        loadingContainer.style.display = 'none';
        // Make sure result is visible
        resultDiv.style.display = 'block';
    }
};

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
    switch(className) {
        case 'lung_n':
            return {
                message: "The tissue appears normal",
                class: "diagnosis-normal"
            };
        case 'lung_aca':
            return {
                message: "Signs of adenocarcinoma detected",
                class: "diagnosis-cancer"
            };
        case 'lung_scc':
            return {
                message: "Signs of squamous cell carcinoma detected",
                class: "diagnosis-cancer"
            };
        default:
            return {
                message: "Unknown tissue type",
                class: ""
            };
    }
}