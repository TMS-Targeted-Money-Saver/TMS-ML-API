from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

app = Flask(_name_)

# Load the TensorFlow H5 model
try:
    model = tf.keras.models.load_model('model/model_user_category (2).h5')  # Path file model H5 Anda
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load dataset untuk fitting vectorizer
try:
    data = pd.read_csv('data/amazon.csv')  # Path file dataset Anda
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File CSV tidak ditemukan. Pastikan path file benar.")
    exit()

# Gabungkan kolom jika semua kolom tersedia
required_columns = ['product_name', 'category', 'about_product']
if all(col in data.columns for col in required_columns):
    data['combined_text'] = data['product_name'] + " " + data['category'] + " " + data['about_product']
    if data['combined_text'].isnull().any():
        print("Error: Null values detected in combined_text. Check your dataset.")
        exit()
else:
    print(f"Error: Dataset is missing one or more required columns: {required_columns}")
    exit()

# Prepare the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
try:
    vectorizer.fit(data['combined_text'])
    print("Vectorizer fitted successfully!")
except Exception as e:
    print(f"Error fitting vectorizer: {e}")
    exit()

# Encode kategori menjadi label numerik
try:
    label_encoder = LabelEncoder()
    data['category_encoded'] = label_encoder.fit_transform(data['category'])
    print("Categories encoded successfully!")
except Exception as e:
    print(f"Error encoding categories: {e}")
    exit()

# Prediction function
def predict_category(text):
    try:
        # Preprocessing input using TfidfVectorizer
        sample_vector = vectorizer.transform([text]).toarray()

        # Debugging: Log the sample vector
        print(f"Input Text: {text}")
        print(f"Vectorized Input: {sample_vector}")

        # Perform prediction
        prediction = model.predict(sample_vector)
        predicted_index = int(np.argmax(prediction))  # Convert to Python int for JSON compatibility

        # Debugging: Log the prediction probabilities
        print(f"Prediction Probabilities: {prediction}")

        # Dapatkan nama kategori dari indeks
        predicted_category = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_category
    except Exception as e:
        return f"Error during prediction: {e}"

# Flask API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        category = predict_category(text)
        if isinstance(category, str) and category.startswith("Error"):
            return jsonify({'error': category}), 500

        return jsonify({'category': category})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if _name_ == '_main_':
    # Set port dynamically using the environment variable, default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
