from ultralytics import YOLO

# LOAD OBB MODEL
model = YOLO("yolov8n-obb.pt")

# TRAIN MODEL
model.train(
    data=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\obb\data.yaml",
    epochs=3,
    imgsz=640
)

# PREDICT
model.predict(
    source=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\obb\test\images",
    save=True
)

print("OBB TASK COMPLETED!")