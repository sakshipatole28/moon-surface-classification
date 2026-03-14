from fastapi import FastAPI, UploadFile, File
from src.pipeline.prediction_pipeline import PredictionPipeline
import shutil
import os

application = FastAPI()

model = PredictionPipeline("artifacts/model.pth")

@application.get("/")
def home():
    return {"message":"Moon Surface Classifier API"}

@application.post("/predict")
async def predict(file: UploadFile = File(...)):

    path = file.filename

    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    prediction,confidence = model.predict(path)

    os.remove(path)

    return {
        "prediction":prediction,
        "confidence":confidence
    }
