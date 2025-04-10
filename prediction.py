import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
import io

class UnknownDiseaseError(Exception):
    """Custom exception for unknown dental diseases"""
    pass

class DentalDiseasePredictor:
    def __init__(self, model_path='dental_model.h5', confidence_threshold=0.3):
        """
        Initialize the predictor with the trained model
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Minimum confidence threshold to accept a prediction
                                If max confidence is below this, assume unknown disease
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224, 3)
        self.confidence_threshold = confidence_threshold
        self.disease_classes = [
            'Caries',
            'Gingivitis',
            'Hypodontia',
            'Mouth Ulcer',
            'Tooth_discoloration'
        ]

    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = load_model(self.model_path)

    def preprocess_image(self, img):
        """Preprocess image for model input"""
        if img is None:
            raise ValueError(f"Failed to process image")
        
        # Preprocess
        img = cv2.resize(img, self.input_shape[:2])
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    def predict(self, image):
        """
        Predict dental disease from image using the trained model
        Raises UnknownDiseaseError if the image likely contains an unknown disease
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Preprocess image
            img = self.preprocess_image(image)

            # Make prediction
            predictions = self.model.predict(img)[0]
            max_confidence = float(np.max(predictions))
            
            # Check if the maximum confidence is below threshold
            if max_confidence < self.confidence_threshold:
                all_probabilities = {
                    disease: float(prob) * 100 
                    for disease, prob in zip(self.disease_classes, predictions)
                }
                raise UnknownDiseaseError(
                    f"Detected possible unknown disease. Confidence too low for known diseases.\n"
                    f"Highest confidence was {max_confidence * 100:.2f}% "
                    f"(threshold: {self.confidence_threshold * 100:.2f}%)\n"
                    f"Probabilities for known diseases:\n"
                    + "\n".join(f"- {disease}: {prob:.2f}%" 
                               for disease, prob in all_probabilities.items())
                )

            predicted_class_index = np.argmax(predictions)

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': self.disease_classes[idx],
                    'confidence': float(predictions[idx]) * 100
                }
                for idx in top_3_indices
            ]

            return {
                'predicted_disease': self.disease_classes[predicted_class_index],
                'confidence': max_confidence * 100,
                'top_3_predictions': top_3_predictions,
                'all_probabilities': {
                    disease: float(prob) * 100 
                    for disease, prob in zip(self.disease_classes, predictions)
                }
            }

        except UnknownDiseaseError:
            raise
        except Exception as e:
            return {'error': str(e)}

def main():
    st.set_page_config(
        page_title="Dental Disease Predictor",
        page_icon="🦷",
        layout="wide"
    )
    
    st.title("🦷 Dental Disease Prediction System")
    st.write("""
    Upload an image of dental condition to get diagnosis predictions.
    This system can identify: Caries, Gingivitis, Hypodontia, Mouth Ulcer, and Tooth discoloration.
    """)
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    model_path = st.sidebar.text_input("Model Path", "dental_model.h5")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.83,
        help="Minimum confidence needed to classify as a known disease"
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a dental image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert PIL Image to OpenCV format
            opencv_image = np.array(image)
            # Convert RGB to BGR (OpenCV format)
            opencv_image = opencv_image[:, :, ::-1].copy() 
        
        with col2:
            if st.button("Predict Disease"):
                with st.spinner("Analyzing image..."):
                    try:
                        predictor = DentalDiseasePredictor(
                            model_path=model_path,
                            confidence_threshold=confidence_threshold
                        )
                        result = predictor.predict(opencv_image)
                        
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            # Display results
                            st.success(f"**Predicted Disease:** {result['predicted_disease']}")
                            st.info(f"**Confidence:** {result['confidence']:.2f}%")
                            
                            # Create a bar chart for all probabilities
                            st.subheader("Disease Probabilities")
                            diseases = list(result['all_probabilities'].keys())
                            probabilities = list(result['all_probabilities'].values())
                            
                            # Sort probabilities for better visualization
                            sorted_indices = np.argsort(probabilities)[::-1]
                            sorted_diseases = [diseases[i] for i in sorted_indices]
                            sorted_probs = [probabilities[i] for i in sorted_indices]
                            
                            chart_data = {
                                'Disease': sorted_diseases,
                                'Probability (%)': sorted_probs
                            }
                            st.bar_chart(chart_data, x='Disease', y='Probability (%)')
                            
                            # Detailed information
                            st.subheader("Detailed Analysis")
                            for disease, prob in sorted(result['all_probabilities'].items(), 
                                                       key=lambda x: x[1], reverse=True):
                                st.write(f"- **{disease}**: {prob:.2f}%")
                    
                    except UnknownDiseaseError as e:
                        st.warning("⚠️ **Unknown Disease Detected**")
                        st.write(str(e))
                    except FileNotFoundError:
                        st.error(f"Model file not found at '{model_path}'. Please check the model path.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    # Information about the diseases
    with st.expander("About Dental Diseases"):
        st.markdown("""
        ### Dental Diseases Information
        
        - **Caries**: Tooth decay that causes cavities
        - **Gingivitis**: Inflammation of the gums
        - **Hypodontia**: Congenital absence of one or more teeth
        - **Mouth Ulcer**: Open sores in the mouth
        - **Tooth Discoloration**: Abnormal coloration of teeth
        
        This system uses a machine learning model trained on dental images to predict these conditions.
        """)
        
    # Instructions for using the app
    with st.expander("How to Use This App"):
        st.markdown("""
        1. Upload a clear image of the dental condition using the file uploader
        2. Click the "Predict Disease" button
        3. View the results and probability distribution
        4. For advanced settings, adjust the confidence threshold in the sidebar
        
        **Note**: This tool is for educational purposes and does not replace professional dental advice.
        """)

if __name__ == "__main__":
    main()
