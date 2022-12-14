import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       'yolov5/runs/train/yolov5s_results/weights/best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)

    cv2.imshow('RESULT', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
