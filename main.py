from ultralytics import YOLO
import cv2
import uuid
import os
from datetime import datetime
from cassandra.cluster import Cluster
import pyttsx3


# ------------------------------
# TEXT-TO-SPEECH
# ------------------------------
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()


# ------------------------------
# CASSANDRA CONNECTION
# ------------------------------
def connect_cassandra():
    try:
        cluster = Cluster(["127.0.0.1"], port=9142)
        session = cluster.connect("fruit_detection")
        print("[OK] Connected to Cassandra!")
        return session
    except Exception as e:
        print("[ERROR] Cassandra connection failed:", e)
        exit()

session = connect_cassandra()


# ------------------------------
# SAVE DETECTION TO DATABASE
# ------------------------------
def save_detection(label, conf, bbox, img_path):
    query = """
    INSERT INTO detections 
    (detection_id, object_type, timestamp, camera_id, confidence, image_path, bbox) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    session.execute(
        query,
        (
            uuid.uuid4(),
            label,
            datetime.now(),
            "CAM-1",
            conf,
            img_path,
            {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
        ),
    )
    print(f"[DB] Saved: {label}, conf={conf:.2f}")


# ------------------------------
# SAVE SNAPSHOT
# ------------------------------
def save_snapshot(frame):
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    path = f"snapshots/{uuid.uuid4()}.jpg"
    cv2.imwrite(path, frame)
    print("[IMG] Saved:", path)
    return path


# ------------------------------
# LOAD HIGH-ACCURACY MODEL
# ------------------------------
model = YOLO("yolov8m.pt")  # Much more accurate than yolov8n
print("[INFO] YOLO model loaded!")


# ------------------------------
# OBJECT CLASSES ALLOWED
# ------------------------------
ALLOWED_OBJECTS = [
    "bottle", "cup", "cell phone", "book", "mouse",
    "banana", "apple", "orange", "remote", "knife",
    "spoon", "fork", "sports ball", "pen", "backpack",
    "laptop", "tv", "keyboard", "charger"
]

# Classes we NEVER detect:
BLOCKED = {"person"}   # Ignore person ALWAYS


print("[INFO] Allowed objects:", ALLOWED_OBJECTS)
print("[INFO] Person detection disabled.")


# ------------------------------
# START CAMERA
# ------------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Detecting objects... Press Q to exit.")

detected_once = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera unavailable.")
        break

    results = model(frame, imgsz=640, verbose=False)

    best_label = None
    best_conf = 0
    best_box = None

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ------------------------------
            # IGNORE PERSON ALWAYS
            # ------------------------------
            if label in BLOCKED:
                continue

            # ------------------------------
            # IGNORE VERY SMALL OBJECTS (nails, fingers)
            # ------------------------------
            width = x2 - x1
            height = y2 - y1
            if width < 60 or height < 60:
                continue

            # ------------------------------
            # ONLY DETECT OUR ALLOWED OBJECTS
            # ------------------------------
            if label not in ALLOWED_OBJECTS:
                continue

            # Confidence threshold
            if conf < 0.50:
                continue

            # Pick object with best confidence
            if conf > best_conf:
                best_label = label
                best_conf = conf
                best_box = (x1, y1, x2, y2)

    # If an allowed object is detected
    if best_label and not detected_once:

        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_label} {best_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        # Say the object name
        speak(f"I detected a {best_label}")

        # Save image + DB entry
        img_path = save_snapshot(frame)
        save_detection(best_label, best_conf, best_box, img_path)

        detected_once = True

        cv2.imshow("Detection", frame)
        cv2.waitKey(1500)
        break

    cv2.imshow("Live View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program finished.")
