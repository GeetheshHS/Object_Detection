from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 nano model (fastest)
model = YOLO("yolov8n.pt")

# Load sample image from Ultralytics
results = model("https://ultralytics.com/images/bus.jpg", show=True)

print("YOLO test completed!")
