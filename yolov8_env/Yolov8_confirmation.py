from ultralytics import YOLO # type: ignore

model = YOLO("yolov8n.pt")  # Load a pretrained model
results = model("C:\\Users\\Dell\\Downloads\\img_6.webp", show=True)  # Run inference
