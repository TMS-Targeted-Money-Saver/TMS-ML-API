import flask
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = flask.Flask(__name__)

# Load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess input
def preprocess_input(text, tokenizer, max_len=5000):
    sequences = tokenizer.texts_to_sequences([text])  # Tokenisasi teks
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')  # Padding ke panjang 5000
    return np.array(padded_sequences, dtype=np.float32)  # Pastikan menjadi float32

# Daftar kategori sesuai dengan model
CATEGORIES = {
    0: "Sports",
    1: "Technology",
    2: "Health",
    3: "Education"
}

# Postprocess output
def postprocess_output(output):
    predicted_class = np.argmax(output)
    predicted_category = CATEGORIES.get(predicted_class, "Unknown")
    return {
        "predicted_class": int(predicted_class),
        "predicted_category": predicted_category
    }

# Load model TFLite
MODEL_PATH = "./model/model_user_category.tflite"
interpreter = load_tflite_model(MODEL_PATH)

# Load tokenizer (tokenizer yang sesuai dengan model Anda)
tokenizer = Tokenizer()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input dari request
        data = flask.request.json
        text_input = data.get("text", "")

        if not text_input:
            return flask.jsonify({"error": "Input text is required."}), 400

        # Preprocessing
        input_data = preprocess_input(text_input, tokenizer)
        
        # Set input tensor
        input_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_index, input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_index = interpreter.get_output_details()[0]['index']
        output_data = interpreter.get_tensor(output_index)

        # Postprocessing
        result = postprocess_output(output_data)

        return flask.jsonify(result), 200
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
