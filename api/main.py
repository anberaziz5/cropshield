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

MODEL = YOLO('best.pt')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    W, H = img.size

    # Run image through YOLOv8 with confidence threshold helper
    results = MODEL(img, conf=0.10)[0]
    
    # 1. Format detections into the exact list of dicts your severity.py expects!
    formatted_detections = []
    if results.boxes is not None:
        for box in results.boxes:
            xyxy_list = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            cls_idx = int(box.cls[0].item())
            cls_name = MODEL.names[cls_idx]
            conf_val = float(box.conf[0].item())
            
            formatted_detections.append({
                'xyxy': xyxy_list,
                'confidence': conf_val,
                'cls_name': cls_name
            })

    # 2. Extract standard flat arrays for your frontend JSON response
    classes = [d['cls_name'] for d in formatted_detections]
    confs = [round(d['confidence'], 3) for d in formatted_detections]

    # 3. Safely call your external model/severity.py functions
    severity = compute_severity(formatted_detections, W, H)
    label = severity_label(severity)

    # Render annotated bounding boxes to send back to the user interface
    ann_img = results.plot()  # numpy BGR image array
    ann_pil = Image.fromarray(ann_img[:,:,::-1])  # Convert BGR to RGB
    buf = io.BytesIO()
    ann_pil.save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'diseases':      classes,
        'confidences':   confs,
        'severity_pct':  severity,
        'severity_label': label,
        'num_detections': len(formatted_detections),
        'annotated_image': f'data:image/jpeg;base64,{img_b64}'
    }

@app.options('/predict')
async def preflight():
    return {'status': 'ok'}

@app.get('/health')
def health(): 
    return {'status':'ok'}