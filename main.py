import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title='CropShield API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your active model weights file
MODEL = YOLO('api/model_v2.pt')

from api.model.severity import compute_severity, severity_label

# 🔍 WORKAROUND: Forcing the home page to display your model classes directly on screen!
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "num_classes": len(MODEL.names),
        "model_classes": MODEL.names
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format received"}

    img_h, img_w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   results = MODEL.predict(img_rgb, conf=0.30, iou=0.45, agnostic_nms=True, device='cpu')[0]

    diseases = []
    confidences = []
    formatted_detections_for_severity = []

    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())
            disease_name = MODEL.names.get(class_id, f"Class_{class_id}")

            # Skip the "healthy" class — it's not a disease, don't count it as severity
            if 'healthy' in disease_name.lower():
                continue

            diseases.append(disease_name)
            confidences.append(confidence_score)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            formatted_detections_for_severity.append({
                'xyxy': [x1, y1, x2, y2]
            })

    if len(diseases) == 0:
        severity_pct = 0.0
        label_string = "Healthy"
    else:
        severity_pct = compute_severity(formatted_detections_for_severity, img_w, img_h)
        label_string = severity_label(severity_pct)

    annotated_img = results.plot()
    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR) if len(results.boxes) > 0 else img
    _, buffer = cv2.imencode('.jpg', annotated_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "severity_pct": float(round(severity_pct, 1)),
        "severity_label": str(label_string),
        "diseases": diseases,
        "confidences": confidences,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }