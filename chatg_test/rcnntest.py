import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Load Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load image and convert to RGB
image_path = r"C:\Users\DELL\Desktop\MajorProject\sampleVideos\Screenshot (479).png"
img = Image.open(image_path).convert("RGB")

# Define image transformations
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

# Convert PIL image to OpenCV format
image_cv = cv2.imread(image_path)

# Process detections
threshold = 0.5  # Confidence threshold
for i, box in enumerate(predictions[0]['boxes']):
    score = predictions[0]['scores'][i].item()
    if score > threshold:
        x1, y1, x2, y2 = map(int, box.tolist())  # Convert to integers
        label = f"{predictions[0]['labels'][i].item()} ({score:.2f})"

        # Draw bounding box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow("Object Detection", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite("detected_objects.jpg", image_cv)
