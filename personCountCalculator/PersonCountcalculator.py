import cv2
import matplotlib.pyplot as plt

def detect_faces_ssd_mobilenet(image_path):
    # Load the pre-trained SSD MobileNet face detection model
    model_path = "deploy.prototxt"  # Model architecture
    weights_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"  # Pre-trained weights
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    # Load the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()
    face_count = 0

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            face_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Faces detected: {face_count}")
    plt.show()

    return face_count

# Test the function
image_path = r"C:\Users\DELL\Desktop\MajorProject\sampleVideos\Screenshot 2024-12-10 173735.png"
number_of_faces = detect_faces_ssd_mobilenet(image_path)
print(f"Number of faces detected: {number_of_faces}")
