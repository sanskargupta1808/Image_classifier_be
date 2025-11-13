import streamlit as st
import sys
import tempfile
import json
from pillow_heif import register_heif_opener

register_heif_opener()
sys.path.append('src')

st.set_page_config(page_title="AI Image Detector API", layout="centered")

# Load detector
@st.cache_resource
def load_detector():
    try:
        from src.detect_hybrid import HybridAIImageDetector
        return HybridAIImageDetector('models', 'three_class_model.pkl')
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None

detector = load_detector()

st.title("🔍 AI Image Detector API")
st.write("Backend service for image analysis")

# API endpoint
uploaded_file = st.file_uploader(
    "Upload image for analysis", 
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic']
)

if uploaded_file and detector:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            result = detector.predict_single(tmp_file.name, return_features=True)
        
        # Display JSON response for API
        st.subheader("API Response:")
        st.json(result)
        
        # Also show formatted result
        st.subheader("Formatted Result:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Classification", result['three_class_prediction'])
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        with col2:
            st.metric("AI Probability", f"{result['ai_probability']:.1%}")
            if result.get('metadata_override'):
                st.success("Camera metadata detected")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

elif not detector:
    st.error("Detector not loaded. Please check model files.")

# Instructions for frontend integration
st.markdown("---")
st.subheader("Integration Instructions")
st.code("""
Frontend Integration:
1. Deploy this Streamlit app
2. Use the app URL as your backend endpoint
3. Send files via the file uploader
4. Parse the JSON response
""")
