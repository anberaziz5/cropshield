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

MODEL = YOLO('api/best.pt')

from api.model.severity import compute_severity, severity_label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format received"}

    img_h, img_w, _ = img.shape

    # 1. FIX: Convert BGR image data to RGB format before sending to YOLOv8
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Run inference on the corrected RGB color space with a low confidence threshold
    results = MODEL.predict(img_rgb, conf=0.10, device='cpu')[0]

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

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            formatted_detections_for_severity.append({
                'xyxy': [x1, y1, x2, y2]
            })

    # 3. Calculate metrics using your logic helpers
    severity_pct = compute_severity(formatted_detections_for_severity, img_w, img_h)
    
    # 4. Fallback: If objects are detected but area math is 0, give it a baseline visibility score
    if len(diseases) > 0 and severity_pct == 0:
        severity_pct = 12.5

    label_string = severity_label(severity_pct)

    # 5. Generate the output visualization layer (using the original canvas to preserve colors)
    annotated_img = results.plot()
    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', annotated_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "severity_pct": float(severity_pct),
        "severity_label": label_string,
        "diseases": diseases,
        "confidences": confidences,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }