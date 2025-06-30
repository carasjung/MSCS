from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import pickle
import librosa
import numpy as np
import os
import warnings
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

model = None
label_encoder = None

def load_model_and_encoder():
    global model, label_encoder
    
    try:
        print("Loading model and encoder...")
        
        model = tf.keras.models.load_model('emotion_detection_model.h5', compile=False)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        print(f"Model loaded successfully!")
        print(f"Label encoder loaded successfully!")
        print(f"Model can classify: {len(label_encoder.classes_)} emotions")
        print(f"Emotions: {', '.join(label_encoder.classes_)}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"Error: Model files not found!")
        print(f"   Make sure you have:")
        print(f"   - emotion_detection_model.h5")
        print(f"   - label_encoder.pkl")
        print(f"   in the same directory as app.py")
        return False
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def create_mel_spectrogram(audio, sr=22050, n_mels=128, hop_length=512):
    """Create mel spectrogram from audio"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def predict_emotion_from_audio(audio_path):
    """Predict emotion from audio file"""
    try:
        audio, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        if len(audio) < sr * 0.5:  # Less than 0.5 seconds
            return {
                'success': False, 
                'error': 'Audio file too short. Please provide audio longer than 0.5 seconds.'
            }
        
        mel_spec = create_mel_spectrogram(audio, n_mels=128)
        
        if mel_spec.shape[1] > 128:
            mel_spec = mel_spec[:, :128]
        else:
            pad_width = 128 - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
        
        prediction = model.predict(mel_spec, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        emotion = label_encoder.inverse_transform([predicted_class])[0]
        
        probabilities = {}
        for i, class_name in enumerate(label_encoder.classes_):
            probabilities[class_name] = float(prediction[0][i])
        
        return {
            'success': True,
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'audio_duration': len(audio) / sr
        }
        
    except Exception as e:
        return {
            'success': False, 
            'error': f'Error processing audio: {str(e)}'
        }

@app.route('/')
def index():
    """Main page"""
    if model is None or label_encoder is None:
        return render_template('error.html', 
                             error="Model not loaded. Please check server logs.")
    return render_template('index.html', emotions=label_encoder.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None or label_encoder is None:
        return jsonify({
            'success': False, 
            'error': 'Model not loaded properly. Please restart the server.'
        })
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'})
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False, 
            'error': f'Unsupported file format. Please use: {", ".join(allowed_extensions)}'
        })
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_emotion_from_audio(filepath)
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None,
        'emotions': list(label_encoder.classes_) if label_encoder else []
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500

if __name__ == '__main__':
    print("🎵 Audio Emotion Detection Server")
    print("=" * 40)
    
    os.makedirs('uploads', exist_ok=True)
    
    if load_model_and_encoder():
        print("Starting server...")
        print("Open your browser and go to: http://127.0.0.1:5000")
        print("Press CTRL+C to stop the server")
        
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("Failed to start server. Please check the error messages above.")
        print("\n Quick fix:")
        print("1. Make sure you've run the Jupyter notebook first to train the model")
        print("2. Check that these files exist in your current directory:")
        print("   - emotion_detection_model.h5")
        print("   - label_encoder.pkl")
        print("3. If files are missing, run the training notebook again")