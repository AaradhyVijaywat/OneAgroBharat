import gzip
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Decompress and save model
with gzip.open("model.pkl.gz", "rb") as f_in:
    model_obj = pickle.load(f_in)
with open("model.pkl", "wb") as f_out:
    pickle.dump(model_obj, f_out)

# Decompress and save preprocessor
with gzip.open("preprocessor.pkl.gz", "rb") as f_in:
    preprocessor_obj = pickle.load(f_in)
with open("preprocessor.pkl", "wb") as f_out:
    pickle.dump(preprocessor_obj, f_out)

print("✅ Decompression complete. Saved as model.pkl and preprocessor.pkl")

# Load uncompressed model and preprocessor
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("preprocessor.pkl", "rb"))

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    year = request.form['Year']
    average_rainfall = request.form['average_rain_fall_mm_per_year']
    pesticides_tonnes = request.form['pesticides_tonnes']
    average_temp = request.form['avg_temp']
    area = request.form['Area']
    item = request.form['Item']

    input_dict = {
        "Year": [year],
        "average_rain_fall_mm_per_year": [average_rainfall],
        "pesticides_tonnes": [pesticides_tonnes],
        "avg_temp": [average_temp],
        "Area": [area],
        "Item": [item]
    }

    input_df = pd.DataFrame(input_dict)

    transformed = scaler.transform(input_df)
    prediction = model.predict(transformed).reshape(-1, 1)

    return render_template('index.html', prediction_text=prediction)

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)