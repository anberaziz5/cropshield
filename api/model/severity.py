# api/model/severity.py
"""
Compute disease severity from YOLOv8 detection results.

Severity is based on the model's own detection confidence for the
diseased class(es) found in the image, expressed as a percentage.
("Healthy" detections are filtered out before this function is ever
called, so an empty `results` list always means a healthy leaf.)
"""

def compute_severity(results, img_w=None, img_h=None):
    if not results:
        return 0.0
    top_conf = max(det['confidence'] for det in results)
    severity = round(top_conf * 100, 1)
    return min(100.0, severity)


def severity_label(pct):
    if pct == 0:    return 'Healthy'
    if pct < 10:    return 'Mild'
    if pct < 30:    return 'Moderate'
    if pct < 60:    return 'Severe'
    return 'Critical'