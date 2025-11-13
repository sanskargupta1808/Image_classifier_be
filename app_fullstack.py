from flask import Flask, request, jsonify, render_template_string
import os
import sys
import tempfile
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()
sys.path.append('src')
from src.detect_hybrid import HybridAIImageDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize detector
try:
    detector = HybridAIImageDetector('models', 'three_class_model.pkl')
    print("✅ Full-stack: Detector loaded successfully")
except Exception as e:
    print(f"❌ Full-stack: Error loading detector: {e}")
    detector = None

# Frontend HTML template
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-area { border: 3px dashed #ddd; padding: 60px 20px; text-align: center; margin: 20px 0; border-radius: 10px; background: white; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover, .upload-area.dragover { border-color: #007bff; background: #f8f9ff; }
        .upload-icon { font-size: 48px; margin-bottom: 20px; color: #007bff; }
        .result { margin-top: 30px; padding: 30px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #007bff; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { color: #dc3545; background: #f8d7da; padding: 15px; border-radius: 5px; }
        .success { color: #155724; background: #d4edda; padding: 15px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; margin-top: 5px; }
        button { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }
        button:hover { background: #0056b3; }
        .image-preview { max-width: 300px; border-radius: 8px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 AI Image Detector</h1>
            <p>Upload an image to detect if it's AI-generated or a real photograph</p>
            <p style="color: #666; font-size: 14px; margin-top: 10px;">
                Full-stack application deployed on Render
            </p>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📁</div>
            <h3>Drag & Drop Image Here</h3>
            <p>or click to browse files</p>
            <p style="color: #666; font-size: 14px; margin-top: 10px;">
                Supports: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP, HEIC
            </p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing your image...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

        async function handleFile(file) {
            if (!file) return;
            
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
            const fileName = file.name.toLowerCase();
            const isValidType = validTypes.includes(file.type) || fileName.includes('.heic') || fileName.includes('.heif');
            
            if (!isValidType) {
                showError('Please upload a valid image file');
                return;
            }
            
            loading.style.display = 'block';
            result.innerHTML = '';
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data, file);
                }
                
            } catch (error) {
                loading.style.display = 'none';
                showError(`Error: ${error.message}`);
            }
        }

        function showResult(data, file) {
            const confidence = (data.confidence * 100).toFixed(1);
            const aiProb = (data.ai_probability * 100).toFixed(1);
            
            const reader = new FileReader();
            reader.onload = function(e) {
                result.innerHTML = `
                    <div class="result">
                        <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 20px;">
                            <img src="${e.target.result}" alt="Analyzed image" class="image-preview">
                            <div style="flex: 1;">
                                <h3>🎯 Analysis Results</h3>
                                <p style="color: #666; margin-bottom: 15px;">File: ${file.name}</p>
                                <div class="metrics">
                                    <div class="metric">
                                        <div class="metric-value">${data.three_class_prediction}</div>
                                        <div class="metric-label">Classification</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-value">${confidence}%</div>
                                        <div class="metric-label">Confidence</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-value">${aiProb}%</div>
                                        <div class="metric-label">AI Probability</div>
                                    </div>
                                </div>
                                ${data.metadata_override ? '<div class="success">✅ Camera metadata detected - Real Photo</div>' : ''}
                            </div>
                        </div>
                        <button onclick="location.reload()">Analyze Another Image</button>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        }

        function showError(message) {
            result.innerHTML = `<div class="result"><div class="error">❌ ${message}</div></div>`;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(FRONTEND_HTML)

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if detector is None:
        return jsonify({'error': 'Detector not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            
            # Validate image
            img = Image.open(tmp_file.name)
            width, height = img.size
            if min(width, height) < 128:
                os.unlink(tmp_file.name)
                return jsonify({'error': 'Image too small (min 128x128)'}), 400
            
            # Get prediction
            result = detector.predict_single(tmp_file.name, return_features=True)
            os.unlink(tmp_file.name)
            
            # Convert numpy types
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                return obj
            
            result = convert_numpy(result)
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
