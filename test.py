from ultralytics import RTDETR
model = RTDETR(
    "../model.pt")
results = model.val(
    split='test',
    data="../DIOR.yaml",
    # data="../NWPU VHR-10.yaml",
    imgsz=640,
    batch=24,
    device=1
    )