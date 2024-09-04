from ultralytics import YOLO

if __name__ == "__main__":
    # If it's your first time change YOLO("yolov8n.yaml")
    model = YOLO("runs/detect/train4/weights/last.pt")
    results = model.train(data="data.yaml", epochs=10, device=0)
