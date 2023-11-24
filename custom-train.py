import cv2

from ultralytics import settings, YOLO

#settings.update({'datasets_dir': '/Users/damien/PycharmProjects/ultralytics/test-files',
#                 'runs_dir':'/Users/damien/PycharmProjects/ultralytics/runs'})

settings.update({'datasets_dir': r'C:\Users\kzammit\Documents\fire-ai\test-files',
                 'runs_dir': r'C:\Users\kzammit\Documents\fire-ai\runs'})

print(settings)

model = YOLO('yolov8-custom.yaml')

results = model.train(data=r'C:\Users\kzammit\Documents\fire-ai\test-files\test.yaml', pretrained=False)

