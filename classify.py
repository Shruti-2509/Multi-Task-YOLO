from ultralytics import YOLO

#  LOAD BASE MODEL
model = YOLO("yolov8n-cls.pt")

# TRAIN MODEL
model.train(
    data=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\classification",
    epochs=3,
    imgsz=224
)


#  LOAD TRAINED MODEL (best.pt)

trained_model = YOLO(
    r"C:\Users\saib9\Desktop\multi_task_yolo\runs\classify\train2\weights\best.pt"
)


# STEP 4: PREDICT 
trained_model.predict(
    source=r"C:\Users\saib9\Desktop\multi_task_yolo\datasets\classification\test\cars_test",
    save=True
)

print("Classification Task Completed Successfully!")