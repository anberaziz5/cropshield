# api/main.py
from model.severity import compute_severity, severity_label
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, base64, numpy as np

# Import the severity logic from your model folder
from model.severity import compute_severity, severity_label

app = FastAPI(title='CropShield API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['*'],
)

# Load your custom trained model weights

MODEL = YOLO('api/best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format"}

    # 1. Force a lower confidence threshold (0.12) so the custom weights register boxes
    results = MODEL.predict(img, conf=0.12, device='cpu')[0]

    diseases = []
    confidences = []

    # 2. Extract bounding boxes using explicit data types safely
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())
            
            # Use names dictionary from your trained model file
            disease_name = MODEL.names.get(class_id, f"Unknown_Issue_{class_id}")
            
            diseases.append(disease_name)
            confidences.append(confidence_score)

    # 3. Import and call your severity metrics logic file
    from api.model.severity import calculate_severity
    severity_pct, severity_label = calculate_severity(diseases, confidences)

    # 4. Generate the annotated image matrix layer with custom overlays drawn
    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "severity_pct": severity_pct,
        "severity_label": severity_label,
        "diseases": diseases,
        "confidences": confidences,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }
    
@app.options('/predict')
async def preflight():
    return {'status': 'ok'}

@app.get('/health')
def health(): 
    return {'status':'ok'}