# Flask ML API Documentation

## Overview
This project provides an API for predicting the category of a product based on textual input using a pre-trained TensorFlow model and scikit-learn for preprocessing.

## Tools and Libraries
### **Backend Framework**
- **Flask**: A lightweight WSGI web application framework for building APIs.

### **Machine Learning Tools**
- **TensorFlow**: For loading and using the pre-trained H5 model.
- **scikit-learn**: For preprocessing text data using TF-IDF vectorization and encoding categories.
- **Pandas**: For dataset loading and manipulation.

### **Utilities**
- **NumPy**: For numerical operations.

### **Data**
- Dataset: `data/amazon.csv` - Used to fit the TF-IDF vectorizer and encode categories.
- Pre-trained Model: `model/model_user_category (2).h5` - TensorFlow model for predictions.

## API Documentation
### Base URL
`http://127.0.0.1:5000`

### Endpoints

#### **1. Predict Category**
Predicts the category of a product based on the input text.

- **URL:**
  `POST /predict`

- **Headers:**
  | Key           | Value       |
  |---------------|-------------|
  | Content-Type  | application/json |

- **Request Body:**
  ```json
  {
      "text": "<product_description>"
  }
  ```

- **Response:**
  - **Success (200):**
    ```json
    {
        "category": "<predicted_category>"
    }
    ```
  
  - **Error (400):**
    ```json
    {
        "error": "No text provided"
    }
    ```

  - **Error (500):**
    ```json
    {
        "error": "<error_details>"
    }
    ```

## Project Structure
```
project/
├── app.py                   # Main Flask app
├── model/
│   └── model_user_category (2).h5  # Pre-trained TensorFlow model
├── data/
│   └── amazon.csv           # Dataset for preprocessing
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

## How to Run the Project
### **1. Install Dependencies**
Create a virtual environment and install the required Python libraries:

```bash
pip install -r requirements.txt
```

### **2. Prepare Dataset and Model**
- Place the dataset file (`amazon.csv`) in the `data/` directory.
- Place the pre-trained model (`model_user_category (2).h5`) in the `model/` directory.

### **3. Run the Flask App**
Start the Flask server:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000/` by default.

### **4. Test the API**
Use tools like **Postman** or **cURL** to test the `/predict` endpoint:

#### Example cURL Command:
```bash
curl -X POST https://tms-ml-api-38627548699.asia-southeast2.run.app/predict \
-H "Content-Type: application/json" \
-d '{"text": "Sample product description"}'
```

### **Expected Output:**
```json
{
    "category": "Predicted Category"
}
```
