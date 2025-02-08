import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (replace with your trained model if available)
model = YOLO("yolov10n.pt")  # Replace with 'your_model.pt' if you have a custom model

# Load image
image_path = r"C:\Users\DELL\Desktop\MajorProject\majorProj_Data\IMG20241224114547.jpg"
image = cv2.imread(image_path)

# Run YOLOv8 inference
results = model(image)
seats = 0
persons = 0
# Process results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0].item()  # Confidence score
        label = result.names[int(box.cls[0].item())]  # Class label

        # Draw bounding box and label
        if label in ["chair", "seat", "person"]:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# count ni of chairs in the label list
        if label == "chair":
            seats += 1
        elif label == "person":
            persons += 1

cv2.putText(image, f"Seats Empty{seats-persons}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3 , (153, 0, 153), 4)

# Resize the image for display if it's too large
max_width = 1200
max_height = 800
height, width = image.shape[:2]

if width > max_width or height > max_height:
    scale = min(max_width / width, max_height / height)
    new_size = (int(width * scale), int(height * scale))
    image_resized = cv2.resize(image, new_size)
else:
    image_resized = image

# Show image with detections
cv2.namedWindow("Seat Detection", cv2.WINDOW_NORMAL)  # Make window resizable
cv2.imshow("Seat Detection", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite("detected_seats.jpg", image)
