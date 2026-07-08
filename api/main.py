import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# 1. Initialize FastAPI app
app = FastAPI(title='CropShield API', version='1.0')

# 2. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load YOLO Model Weights
MODEL = YOLO('api/best.pt')

# 4. Import both function keys from your severity file
from api.model.severity import compute_severity, severity_label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format received"}

    # Get image dimensions dynamically for your area formula
    img_h, img_w, _ = img.shape

    # Run inference with lower threshold (0.12) to catch custom classes
    results = MODEL.predict(img, conf=0.12, device='cpu')[0]

    diseases = []
    confidences = []
    formatted_detections_for_severity = []

    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())
            disease_name = MODEL.names.get(class_id, f"Unknown_Issue_{class_id}")
            
            diseases.append(disease_name)
            confidences.append(confidence_score)

            # Extract raw xyxy coordinates for the severity function
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            formatted_detections_for_severity.append({
                'xyxy': [x1, y1, x2, y2]
            })

    # Calculate severity percentage and label using your exact functions!
    severity_pct = compute_severity(formatted_detections_for_severity, img_w, img_h)
    label_string = severity_label(severity_pct)

    # If the model found diseases but they took up 0% of the area mathematically, 
    # force it out of 'Healthy' so the UI reveals the name correctly.
    if len(diseases) > 0 and severity_pct == 0:
        severity_pct = 5.0
        label_string = "Mild"

    # Generate the annotated image matrix layer with bounding boxes drawn
    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "severity_pct": severity_pct,
        "severity_label": label_string,
        "diseases": diseases,
        "confidences": confidences,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }