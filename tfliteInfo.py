import tensorflow as tf

# Memuat model TFLite
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Memeriksa informasi tentang input dan output tensor
def get_model_info(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input Details:", input_details)
    print("Output Details:", output_details)

# Path ke model .tflite
MODEL_PATH = "./model/model_user_category.tflite"
interpreter = load_tflite_model(MODEL_PATH)

# Cek informasi input dan output
get_model_info(interpreter)
