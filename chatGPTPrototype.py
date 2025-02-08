import cv2
import torch
from ultralytics import YOLO  # For YOLOv8

# Load the YOLO model (fine-tuned for seat detection)
model = YOLO(r"yolov8n.pt")

# Mapping seat IDs (row, column) to seat numbers
seat_map = {(0, 0): "A1", (0, 1): "A2"}  # Example

# Open video stream
cap = cv2.VideoCapture("auditorium_feed.mp4")  # Replace with 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO model
    results = model(frame)
    empty_seats = []

    # Process results
    for result in results.xyxy[0]:  # Bounding boxes
        x1, y1, x2, y2, conf, cls = result
        if cls == 1:  # Assuming '1' is the class for empty seats
            seat_id = ...  # Calculate seat ID using x1, y1, etc.
            seat_number = seat_map.get(seat_id)
            empty_seats.append(seat_number)

            # Display bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display results
    print(f"Empty seats: {empty_seats}")
    cv2.imshow("Frame", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
