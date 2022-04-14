import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3001",
    "http://localhost:3000",
    "http://localhost:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8502/v1/models/potatoes_model:predict"

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello, from Jayshil!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)

    json_data={
        "instances": img_batch.tolist()
    }

    response=requests.post(endpoint,json=json_data)
    prediction=np.array(response.json()["predictions"][0])

    predicted_class=CLASS_NAMES[np.argmax(prediction)]
    confidence=np.max(prediction)

    return {
        "class":predicted_class,
        "confidence":float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

    #  docker run -t --rm -p 8502:8502 -v C:/Code/potato-disease:/potato-disease tensorflow/serving --rest_api_port=8502 --model_config_file=/potato-disease/models.config