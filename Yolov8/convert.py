from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")  #for best.pt

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'   Optional:also create first best.onnx then best.engine

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")    #for best.engine

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")




# from ultralytics import YOLO

# # Load the YOLO model
# model = YOLO("yolo11n.pt")

# # Export to TensorRT INT8 format using calibration dataset
# model.export(
#     format="engine", 
#     int8=True,  # Enable INT8 quantization
#     device=0,  # Use GPU
#     calibration_images="/ultralytics/calibration_images"  # Folder with calibration images
# )
