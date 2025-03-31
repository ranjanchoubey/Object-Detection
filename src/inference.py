import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
model_loaded = YOLO(model_path)

def predict(image):
    # Read the image from the file
    image_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Perform inference with labels
    results = model_loaded(image, conf=0.50)[0]

    # Create Detections with labels
    detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )

    # Get class names from the model
    class_names = model_loaded.names

    # Annotate boxes and labels with class names
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,  # Adjust text size if needed
        text_thickness=1,  # Adjust text thickness  # You can customize color palette
    )

    # Annotate the image with detections and their labels
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=[f"{class_names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    )

    # Display annotated image
    _, buffer = cv2.imencode('.jpg', annotated_image)

    # Print detection details
    results = []
    for box, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
        results.append([f"Detection: {class_names[cls]}, Confidence: {conf:.2f}, Coordinates: {box}"])

    return buffer, results