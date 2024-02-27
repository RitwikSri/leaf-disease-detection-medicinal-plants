from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['Alstonia Scholaris Diseased: Foliar Galls',
 'Alstonia Scholaris Healthy',
 'Arjun Diseased: Leaf Spot',
 'Arjun Healthy',
 'Bael Diseased: Chlorosis',
 'Basil Healthy',
 'Chinar Diseased: Leaf Spot',
 'Chinar Healthy',
 'Guava Diseased: Fungal Disease',
 'Guava Healthy',
 'Jamun Diseased: Fungal Disease',
 'Jamun Healthy',
 'Jatropha Diseased: Leaf Spot',
 'Jatropha Healthy',
 'Lemon Diseased: Citrus Canker',
 'Lemon Healthy',
 'Mango Diseased: Anthracnose',
 'Mango Healthy',
 'Pomegranate Diseased: Cercospora Spot',
 'Pomegranate Healthy',
 'Pongamia Pinnata Diseased: Chlorotic Lesions',
 'Pongamia Pinnata Healthy']

MODEL = tf.keras.models.load_model("D:/Misc/leaf-disease-detection/models/2")

@app.get("/ping")
async def ping():
    return "Hello, I am alive" 

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    #resize image
    #newsize = (320, 480)
    #image = image.resize(newsize)

    img_batch = np.expand_dims(image, 0)


    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)