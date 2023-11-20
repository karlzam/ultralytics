import cv2

from ultralytics import settings, YOLO

#settings.update({'datasets_dir': '/Users/damien/PycharmProjects/ultralytics/test-files',
#                 'runs_dir':'/Users/damien/PycharmProjects/ultralytics/runs'})

settings.update({'datasets_dir': r'C:\Users\Karlee\Documents\FireAI\test-files',
                 'runs_dir': r'C:\Users\Karlee\Documents\FireAI\runs'})

print(settings)

model = YOLO('yolov8x.pt')

#sample_image = cv2.imread("test-files/train/images/ecce-c20d-dji_0005.1664412091751.1675213808715.tiff",
#                          cv2.IMREAD_UNCHANGED)

#sample_image = cv2.imread(r"C:\Users\Karlee\Documents\FireAI\test-files\train\images\ecce-c20d-dji_0005.1664412091751.1675213808715.tiff",
#                          cv2.IMREAD_UNCHANGED)

#sample_image = cv2.imread(r"C:\Users\Karlee\Documents\FireAI\test-files\train\images\ecce-c20d-dji_0005.1664412091751"
#                          r".1675213808715.tiff")

#print(sample_image.dtype)

results = model.train(data='test-files/test.yaml')

