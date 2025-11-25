import sys
sys.path.append('../')
from ultralytics import RTDETR

model = RTDETR("../KARFDet.yaml")

model.train(
             data="../DIOR.yaml",
　　　　　　# data="../NWPU VHR-10.yaml",
            optimizer="AdamW",
            epochs=300,
            iflrAuto=True,
            lr0 = 0.001,
            patience=30,
            imgsz=640,
            batch=24,
            device=0
            )
