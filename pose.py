from ultralytics import YOLO

# LOAD POSE MODEL
model = YOLO("yolov8n-pose.pt")

# TRAIN MODEL
model.train(
    data=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\pose\data.yaml",
    epochs=3,
    imgsz=640
)

# PREDICT 
model.predict(
    source=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\pose\test\images",
    save=True
)

# WEBCAM 
model = YOLO(r"C:\Users\saib9\Desktop\multi_task_yolo\runs\pose\train\weights\best.pt")

model.predict(source=0, show=True)

print("Pose Estimation Completed!")