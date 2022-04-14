# import numpy as np
# from fastapi import FastAPI, UploadFile, File
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# endpoint = "http://localhost:8505/v1/models/potatoes_model:predict"
# # MODEL = tf.keras.models.load_model("../saved_models/1")
# CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch=np.expand_dims(image,0)
#     preditciton=MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(preditciton[0])]
#     confidence = np.max(preditciton[0])
#     return {
#         'class':predicted_class,
#         'confidence':float(confidence)
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)