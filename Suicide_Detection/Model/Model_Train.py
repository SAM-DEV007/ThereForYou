from ultralytics import YOLO
from pathlib import Path

import os


if __name__ == '__main__':
    config_path = Path(__file__).resolve().parent / 'Model_Data/config.yaml'
    save_path = Path(__file__).resolve().parent / 'Model_Data/logs'

    os.makedirs(str(save_path), exist_ok=True)

    if not os.path.exists(str(save_path / 'runs/detect/SuicideDetection')):
        model = YOLO('yolov8n.yaml')
        model.to('cuda')

        model.train(data=str(config_path), imgsz=640, batch=10, epochs=50, name='SuicideDetection', workers=3, save_dir=str(save_path), amp=False)
    else:
        model = YOLO(str(save_path / 'runs/detect/SuicideDetection/weights/last.pt'))
        model.to('cuda')

        model.train(resume=True, data=str(config_path), imgsz=640, batch=10, epochs=60, name='SuicideDetection', workers=3, save_dir=str(save_path), amp=False)
    
    os.remove(str(Path.cwd() / 'yolov8n.pt'))