from ultralytics import YOLO

# Load YOLO model (nano lightweight)
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\detection\data.yaml",
    epochs=5,
    imgsz=640
)

# Predict on test images
model.predict(
    source=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\detection\test\images",
    save=True
)

# Webcam deployment (IMPORTANT for assignment)
model = YOLO(r"C:\Users\saib9\Desktop\multi_task_yolo\runs\detect\train\weights\best.pt")
model.predict(source=0)