"""
Breast Cancer Prediction System - Flask Web Application

DISCLAIMER: This system is strictly for educational purposes and must not be used as a medical diagnostic tool.

This Flask application loads a pre-trained machine learning model and provides a web interface
for predicting breast cancer diagnosis (Benign/Malignant) based on user input.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sys

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/breast_cancer_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
SELECTED_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
DIAGNOSIS_LABELS = {0: 'Benign', 1: 'Malignant'}

# Global variables for model and scaler
model = None
scaler = None


def load_model_and_scaler():
    """
    Load the pre-trained model and scaler from disk.
    
    Returns:
        tuple: (model, scaler) or (None, None) if loading fails
    """
    global model, scaler
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            return None, None
        
        if not os.path.exists(SCALER_PATH):
            print(f"ERROR: Scaler file not found at {SCALER_PATH}")
            return None, None
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        print("✓ Model and scaler loaded successfully!")
        return model, scaler
    
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return None, None


@app.route('/')
def index():
    """Render the main prediction page."""
    if model is None or scaler is None:
        return "Error: Model not loaded. Please ensure model files exist.", 500
    
    return render_template('index.html', features=SELECTED_FEATURES)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the web interface.
    
    Expected JSON format:
    {
        "radius_mean": float,
        "texture_mean": float,
        "perimeter_mean": float,
        "area_mean": float,
        "smoothness_mean": float
    }
    
    Returns:
        JSON response with prediction and confidence
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Extract feature values
        feature_values = []
        for feature in SELECTED_FEATURES:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            
            try:
                value = float(data[feature])
                if value < 0:
                    return jsonify({'error': f'{feature} must be non-negative'}), 400
                feature_values.append(value)
            except (ValueError, TypeError):
                return jsonify({'error': f'{feature} must be a valid number'}), 400
        
        # Convert to numpy array and scale
        input_array = np.array([feature_values])
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Prepare response
        diagnosis = DIAGNOSIS_LABELS[prediction]
        confidence = float(max(prediction_proba) * 100)
        benign_prob = float(prediction_proba[0] * 100)
        malignant_prob = float(prediction_proba[1] * 100)
        
        response = {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob,
            'input_features': {feature: feature_values[i] for i, feature in enumerate(SELECTED_FEATURES)},
            'disclaimer': 'This system is strictly for educational purposes and must not be used as a medical diagnostic tool.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/features', methods=['GET'])
def get_features():
    """Return information about the selected features."""
    features_info = {
        'radius_mean': {
            'name': 'Radius Mean',
            'description': 'Mean of distances from center to points on the perimeter',
            'unit': 'micrometers'
        },
        'texture_mean': {
            'name': 'Texture Mean',
            'description': 'Standard deviation of gray-scale values',
            'unit': 'N/A'
        },
        'perimeter_mean': {
            'name': 'Perimeter Mean',
            'description': 'Mean size of the core tumor',
            'unit': 'micrometers'
        },
        'area_mean': {
            'name': 'Area Mean',
            'description': 'Mean area of the nuclei',
            'unit': 'square micrometers'
        },
        'smoothness_mean': {
            'name': 'Smoothness Mean',
            'description': 'Local variation in radius lengths',
            'unit': 'N/A'
        }
    }
    
    return jsonify({
        'features': SELECTED_FEATURES,
        'details': features_info,
        'algorithm': 'Logistic Regression',
        'dataset': 'Breast Cancer Wisconsin (Diagnostic)',
        'model_path': MODEL_PATH,
        'scaler_path': SCALER_PATH
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information and statistics."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        model_info_data = {
            'algorithm': type(model).__name__,
            'features_used': SELECTED_FEATURES,
            'number_of_features': len(SELECTED_FEATURES),
            'model_parameters': {
                'random_state': 42,
                'max_iter': 1000,
                'solver': 'lbfgs'
            },
            'classes': list(DIAGNOSIS_LABELS.values()),
            'model_size_kb': os.path.getsize(MODEL_PATH) / 1024 if os.path.exists(MODEL_PATH) else None,
            'disclaimer': 'This system is strictly for educational purposes and must not be used as a medical diagnostic tool.'
        }
        return jsonify(model_info_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load model on startup
    print("Loading model and scaler...")
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        print("CRITICAL: Model loading failed!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Breast Cancer Prediction System - Flask Server")
    print("="*60)
    print(f"Model: {type(model).__name__}")
    print(f"Features: {', '.join(SELECTED_FEATURES)}")
    print(f"Server starting at http://localhost:5000")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
