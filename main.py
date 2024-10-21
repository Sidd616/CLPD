from ultralytics import YOLO
import cv2
import numpy as np
import csv
from sort.sort import *
from util import get_car, read_license_plate

# Dictionary to store results
results = {}

# Initialize the SORT tracker for vehicle tracking
mot_tracker = Sort()

# Load YOLO models
try:
    coco_model = YOLO('yolov8n.pt')  # COCO model for detecting vehicles
    license_plate_detector = YOLO('models/best.pt')  # Custom license plate model
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit()

# Load the image
image_path = 'image.jpg'  # Change this to your image file
frame = cv2.imread(image_path)

# Check if the image was opened successfully
if frame is None:
    print("Error: Could not open image.")
    exit()

print("Processing image")

# Detect vehicles in the frame
detections = coco_model(frame)[0]
detections_ = []
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in [2, 3, 5, 7]:  # Only vehicle classes
        detections_.append([x1, y1, x2, y2, score])

print(f"Detected vehicles: {len(detections_)}")

# Track vehicles
track_ids = mot_tracker.update(np.asarray(detections_))

# Detect license plates in the frame
license_plates = license_plate_detector(frame)[0]
print(f"Detected license plates: {len(license_plates.boxes.data)}")

# Store results in a list for writing to CSV later
csv_rows = []

for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

    # Assign license plate to car using the tracker's IDs
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

    # Check if the license plate was assigned to a car
    if car_id != -1:
        # Crop the license plate from the frame
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Convert the crop to grayscale and apply thresholding
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        try:
            # Read the license plate text using OCR
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            # Store the result if the text is detected
            if license_plate_text is not None:
                results[car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }
                csv_rows.append({
                    'frame_nmr': 0,  # Only one image, so frame_nmr is 0
                    'car_id': car_id,
                    'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'license_plate_bbox_score': score,
                    'license_number': license_plate_text,
                    'license_number_score': license_plate_text_score,
                })
                print(f"License plate detected for car {car_id}: {license_plate_text}")
        except Exception as e:
            print(f"Error reading license plate: {e}")

# Write results to a CSV file once after processing the image
try:
    with open('test.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score'])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print("Results saved to test.csv")
except Exception as e:
    print(f"Error writing results to CSV: {e}")

# Optionally, you can show the image with detections
cv2.imshow("Detected Vehicles and License Plates", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
