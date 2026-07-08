# model/severity.py
"""
Compute disease severity from YOLOv8 detection results.
Severity = (sum of bounding box areas) / (total image area) * 100
"""

def compute_severity(results, img_w, img_h):
    """
    results: list of dicts with keys: xyxy (x1,y1,x2,y2), conf, cls_name
    Returns: severity percentage (0-100)
    """
    total_img_area = img_w * img_h
    disease_area   = 0

    for det in results:
        x1, y1, x2, y2 = det['xyxy']
        box_area = (x2-x1) * (y2-y1)
        disease_area += box_area

    # Cap at 100%
    severity = min(100.0, (disease_area / total_img_area) * 100)
    return round(severity, 1)

def severity_label(pct):
    if pct == 0:    return 'Healthy'
    if pct < 10:    return 'Mild'
    if pct < 30:    return 'Moderate'
    if pct < 60:    return 'Severe'
    return 'Critical'