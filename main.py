import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from activities_logics import MoondreamService

load_dotenv()

app = FastAPI()

@app.post("/detect-activities/")
def detect_activities(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    try:
        contents = file.file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        image = Image.open(temp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Detect activities
    service = MoondreamService()
    detections = service.detect_activities(image)

    # Clean up temp file
    os.remove(temp_path)

    return JSONResponse(content={
        "detections": detections
    }) 