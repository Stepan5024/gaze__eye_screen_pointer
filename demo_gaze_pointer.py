from l2cs import Pipeline, render
import cv2
import torch
import pyautogui as pag
import pathlib

CWD = pathlib.Path.cwd()

if torch.cuda.is_available():
    gaze_pipeline = Pipeline(
        weights= CWD / 'models' / 'L2CSNet' / 'Gaze360' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cuda')
    )
else:
    gaze_pipeline = Pipeline(
        weights= CWD / 'models' / 'L2CSNet' / 'Gaze360' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cpu')
    )
cam_id = 0
cap = cv2.VideoCapture(cam_id)
while True:
    show, frame = cap.read()

    results = gaze_pipeline.step(frame)
    frame, dx, dy, h, w = render(frame, results)
    h, w, _ = frame.shape

    dx += w / 2
    dy += h / 2

    width, height = pag.size()

    mouse_x = width - (width / 2 + dx * width / w * 4)
    mouse_y = dy * height / h * 4

    if (0 < mouse_x < width and 0 < mouse_y < height):
        pag.moveTo(mouse_x, mouse_y)



