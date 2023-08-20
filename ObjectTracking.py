import cv2
import numpy as np
import time
import torch
from djitellopy import Tello
from ultralytics import YOLO

host = ''
port = 9000
local_address = (host, port)

# Set points (center of the frame coordinates in pixels)
refX = 960 / 2
refY = 720 / 2

# PI constants
Kp_X = 0.1
Ki_X = 0.0
Kp_Y = 0.2
Ki_Y = 0.0

# Loop time
Tc = 0.05

# PI terms initialized
integral_X = 0
error_X = 0
previous_error_X = 0
integral_Y = 0
error_Y = 0
previous_error_Y = 0

centroX_prev = refX
centroY_prev = refY

# Load the YOLOv5 model
model = YOLO('yolov8n.pt')

# Drone initialization
drone = Tello()
time.sleep(2)
print("Connecting...")
drone.connect()
print("BATTERY: ", drone.get_battery())
time.sleep(1)
print("Takeoff...")
drone.streamon()
drone.takeoff()

while True:
    start = time.time()
    frame = drone.get_frame_read().frame

    cv2.circle(frame, (int(refX), int(refY)), 1, (0, 0, 255), 10)

    h, w, _ = frame.shape

    # Perform object detection using YOLOv5
    results = model(frame)
    detections = results.pred[0]
    for detection in detections:
        label, confidence, bbox = detection[5], detection[4], detection[:4]
        if label == 0 and confidence > 0.5:
            startX, startY, endX, endY = map(int, bbox)
            centroX = (startX + endX) / 2
            centroY = (startY + endY) / 2

            centroX_prev = centroX
            centroY_prev = centroY

            cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

            error_X = -(refX - centroX)
            error_Y = refY - centroY

            cv2.line(frame, (int(refX), int(refY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

            # PI controller
            integral_X += error_X * Tc
            uX = Kp_X * error_X + Ki_X * integral_X
            previous_error_X = error_X
            integral_Y += error_Y * Tc
            uY = Kp_Y * error_Y + Ki_Y * integral_Y
            previous_error_Y = error_Y
            drone.send_rc_control(0, 0, round(uY), round(uX))

            break

    else:  # If nobody is recognized, use previous frame's centerX and centerY as reference
        centroX = centroX_prev
        centroY = centroY_prev
        cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)
        error_X = -(refX - centroX)
        error_Y = refY - centroY

        cv2.line(frame, (int(refX), int(refY)), (int(centroX), int(centroY)), (0, 255, 255), 5)
        integral_X += error_X * Tc
        uX = Kp_X * error_X + Ki_X * integral_X
        previous_error_X = error_X
        integral_Y += error_Y * Tc
        uY = Kp_Y * error_Y + Ki_Y * integral_Y
        previous_error_Y = error_Y
        drone.send_rc_control(0, 0, round(uY), round(uX))
        continue

    cv2.imshow("Frame", frame)
    end = time.time()
    elapsed = end - start
    if Tc - elapsed > 0:
        time.sleep(Tc - elapsed)
    end_ = time.time()
    elapsed_ = end_ - start
    fps = 1 / elapsed_
    print("FPS: ", fps)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

drone.streamoff()
cv2.destroyAllWindows()
print("Landing...")
drone.land()
print("BATTERY: ", drone.get_battery())
drone.end()