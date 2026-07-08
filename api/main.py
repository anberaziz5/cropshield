# api/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, base64, numpy as np

app = FastAPI(title='CropShield API', version='1.0')

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Load your custom trained model weights
MODEL = YOLO('best.pt')

def compute_severity(boxes, img_w, img_h):
    # Severity = (sum of bounding box areas) / (total image area) * 100
    area = sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    return round(min(100.0, area/(img_w*img_h)*100), 1)

def severity_label(pct):
    if pct == 0:  return 'Healthy'
    if pct < 10:  return 'Mild'
    if pct < 30:  return 'Moderate'
    if pct < 60:  return 'Severe'
    return 'Critical'

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    W, H = img.size

    # Run image through YOLOv8
    results   = MODEL(img)[0]
    boxes     = [b.xyxy[0].tolist() for b in results.boxes]
    classes   = [MODEL.names[int(b.cls)] for b in results.boxes]
    confs     = [round(float(b.conf),3) for b in results.boxes]
    severity  = compute_severity(boxes, W, H)

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
        'severity_label': severity_label(severity),
        'num_detections': len(boxes),
        'annotated_image': f'data:image/jpeg;base64,{img_b64}'
    }

@app.get('/health')
def health(): 
    return {'status':'ok'}