# app_flask.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("cat_vs_dog_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file.save("temp.jpg")

    img = image.load_img("temp.jpg", target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    result = "dog" if prob > 0.5 else "cat"

    return jsonify({"prediction": result, "probability": float(prob)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)