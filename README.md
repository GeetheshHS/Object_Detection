Handheld Object Detection Using YOLOv8 + Python + Cassandra (Docker)

This project is a real-time object detection system that identifies objects held in a user’s hand using a webcam.
It uses the YOLOv8 model for fast and accurate detection and stores every detection event in a Cassandra NoSQL database running inside Docker.

Unlike normal detection systems, this project is optimized to:

✔ Ignore people (no person detection)
✔ Only detect objects that the user is holding
✔ Avoid small or irrelevant objects (nails, fingers, shadows, etc.)
✔ Provide Audio Feedback using Text-to-Speech (TTS)
✔ Save each detection as an image snapshot
✔ Store structured detection information in Cassandra


🔍 Features
🎥 Real-Time Object Detection

Uses YOLOv8 (yolov8m.pt or custom-trained model)

Detects only selected objects (bottle, phone, book, laptop charger, pen, spoon, mouse, etc.)

Ignores people completely to avoid false triggers.

🤖 Handheld Object Focus

Filters small bounding boxes to avoid detecting hands, nails, or background items.

Ensures the dominant object in hand is the one detected.

🔊 Text-to-Speech Output

Announces the detected object (e.g., “I detected a bottle”).

Makes the system usable for visually impaired individuals or hands-free applications.

📸 Auto Image Snapshot

Captures the frame whenever a valid object is detected.

Stores snapshots in the /snapshots directory.

🗄️ Cassandra NoSQL Storage

Stores each detection with:
UUID
Object type
Confidence score
Bounding box coordinates
Snapshot path
Timestamp
Camera ID

🐳 Docker-Based Cassandra

Cassandra runs inside a Docker container
No installation needed on the host machine
Easy to reset and manage



🧱 Tech Stack
Component	Technology
Object Detection	YOLOv8 (Ultralytics)
Programming Language	Python
Database	Apache Cassandra
Database Deployment	Docker
Speech Output	pyttsx3
Image Processing	OpenCV
UUID Handling	Python UUID
Time Stamps	datetime
📁 Project Workflow

Start the Cassandra container in Docker

Run the Python script (main.py)
YOLO reads frames from the webcam
Object detected → filters only allowed objects
If valid detection:
Draw bounding box
Speak object name
Save snapshot
Insert record into Cassandra
Exit after first detection
User can restart detection anytime
