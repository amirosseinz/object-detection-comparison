"""Validate the TensorFlow Serving response for one object-detection image."""
import math


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def parse_detections(result, confidence_threshold):
    """Return validated (box, score, class) tuples above the confidence threshold."""
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError("Model response must contain a successful prediction object.")
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], dict):
        raise ValueError("Expected exactly one prediction object for the input image.")
    prediction = predictions[0]
    arrays = [prediction.get(name) for name in ("detection_boxes", "detection_scores", "detection_classes")]
    if not all(isinstance(values, list) for values in arrays):
        raise ValueError("Prediction requires detection_boxes, detection_scores and detection_classes arrays.")
    boxes, scores, classes = arrays
    if len(boxes) != len(scores) or len(boxes) != len(classes):
        raise ValueError("Detection arrays must have equal lengths.")
    count = prediction.get("num_detections", len(boxes))
    if not finite_number(count) or int(count) != count or not 0 <= count <= len(boxes):
        raise ValueError("num_detections must be an integer within the detection array length.")

    detections = []
    for index in range(int(count)):
        box, score, class_id = boxes[index], scores[index], classes[index]
        if not isinstance(box, list) or len(box) != 4 or not all(finite_number(value) for value in box):
            raise ValueError(f"Detection {index} requires four finite box coordinates.")
        ymin, xmin, ymax, xmax = box
        if ymin > ymax or xmin > xmax:
            raise ValueError(f"Detection {index} has reversed box coordinates.")
        if not finite_number(score) or not 0 <= score <= 1:
            raise ValueError(f"Detection {index} requires a finite probability score.")
        if not finite_number(class_id) or class_id < 0 or int(class_id) != class_id:
            raise ValueError(f"Detection {index} requires a nonnegative integer class ID.")
        if score > confidence_threshold:
            detections.append((box, score, int(class_id)))
    return detections
