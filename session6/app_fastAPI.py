# app_fastapi.py
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = FastAPI()
model = load_model("cat_vs_dog_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as f:
        f.write(await file.read())

    img = image.load_img("temp.jpg", target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    result = "dog" if prob > 0.5 else "cat"

    return {"prediction": result, "probability": float(prob)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)