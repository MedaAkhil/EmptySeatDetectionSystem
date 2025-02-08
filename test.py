from flask import Flask, jsonify
from flask_cors import CORS
import time
import cv2
import torch
from ultralytics import YOLO


model = YOLO("yolov10n.pt")



app = Flask(__name__)
CORS(app)  # Enable CORS for all requests

@app.route('/getdata', methods=['GET'])
def get_data():


    # image_path = r"C:\Users\DELL\Desktop\MajorProject\majorProj_Data\IMG20241224114547.jpg"
    # image_path = r"C:\Users\DELL\Desktop\MajorProject\sampleVideos\0affaab53fee437f301200601e1ca08a.jpg"
    # image_path = r"C:\Users\DELL\Desktop\MajorProject\sampleVideos\Muzo-tables_kids_working-450x300.jpg"
    image_path = r"C:\Users\DELL\Desktop\MajorProject\sampleVideos\d-1000x1000.jpg"
    image = cv2.imread(image_path)

    results = model(image)
    seats = 0
    persons = 0
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

    return jsonify({
        "seatsAvailable": seats-persons,
        "totalSeats": seats,
        "SpaceNumber": "1",
        "timestamp": int(time.time() * 1000)  # Equivalent to JavaScript's Date.now()
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
