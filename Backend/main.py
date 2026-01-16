import uvicorn
from fastapi import FastAPI, File, UploadFile
from models.efficientnet import load_models, get_model
from utils.preprocess import preprocess_image
import numpy as np

from reasoning.schemas import SymptomInput
from reasoning.llm_reasoner import build_llm_prompt, call_llm


app = FastAPI(title="Disease Detection API")

@app.on_event("startup")
def startup_event():
    load_models()
    print("FastAPI startup complete")

@app.get("/health")
def health_check():
    return {"status": "API is running", "models_loaded": True}    

CLASSES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image
    image = preprocess_image(file.file)

    # 2. Get model
    model = get_model("eye")

    # 3. Run prediction
    preds = model.predict(image)[0]

    # 4. Get class + confidence
    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])

    return {
        "prediction": CLASSES[class_index],
        "confidence": round(confidence, 4)
    }

@app.post("/predict/symptoms")
async def predict_symptoms(data: SymptomInput):
    prompt = build_llm_prompt(data)
    analysis = call_llm(prompt)

    return {"analysis": analysis}

   
