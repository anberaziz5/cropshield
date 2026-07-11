# model/severity.py
import numpy as np
"""
Compute disease severity from YOLOv8 detection results.
Severity = (sum of bounding box areas) / (total image area) * 100
"""

def compute_severity(results, img_w, img_h):
    if not results:
        return 0.0

    mask = np.zeros((img_h, img_w), dtype=bool)
    for det in results:
        x1, y1, x2, y2 = det['xyxy']
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        mask[y1:y2, x1:x2] = True

    severity = (mask.sum() / (img_w * img_h)) * 100
    return round(min(100.0, severity), 1)

def severity_label(pct):
    if pct == 0:    return 'Healthy'
    if pct < 10:    return 'Mild'
    if pct < 30:    return 'Moderate'
    if pct < 60:    return 'Severe'
    return 'Critical'