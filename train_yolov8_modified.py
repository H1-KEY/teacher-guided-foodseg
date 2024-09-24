from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")
model.train(
    data="./food-seg.yaml",
    cfg="./yolov8_modified.yaml",
)
