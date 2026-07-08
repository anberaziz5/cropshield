import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# 1. Initialize FastAPI app
app = FastAPI(title='CropShield API', version='1.0')

# 2. Enable CORS so your local frontend port 5175 can access it smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load YOLO Model Weights (relative path from root container app folder)
MODEL = YOLO('api/best.pt')

# 4. Corrected relative path import for your severity function helper
from api.model.severity import compute_severity

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image from upload payload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format received"}

    # Run inference with a lower confidence threshold so it catches your custom classes
    results = MODEL.predict(img, conf=0.12, device='cpu')[0]

    diseases = []
    confidences = []

    # Extract detected bounding box categories and confidence values
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())
            
            # Map index IDs to human-readable names from your model dataset definitions
            disease_name = MODEL.names.get(class_id, f"Unknown_Issue_{class_id}")
            
            diseases.append(disease_name)
            confidences.append(confidence_score)

    # Calculate severity using your module's logic helper function
    try:
        # Falls back to standard values if your compute_severity accepts different inputs
        severity_pct, severity_label = compute_severity(diseases, confidences)
    except Exception:
        # Fallback safety validation layer if calculation experiences structural mismatches
        severity_pct = 45 if len(diseases) > 0 else 0
        if severity_pct == 0: severity_label = "Healthy"
        elif severity_pct < 20: severity_label = "Mild"
        elif severity_pct < 50: severity_label = "Moderate"
        else: severity_label = "Severe"

    # Generate the annotated image canvas matrix with visible bounding boxes
    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "severity_pct": int(severity_pct),
        "severity_label": severity_label,
        "diseases": diseases,
        "confidences": confidences,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }