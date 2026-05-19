import cv2
import sys
import os
import re
import time
import csv
from ultralytics import YOLO
import easyocr

# -----------------------------
# LOAD MODELS
# -----------------------------
model = YOLO("models/license_plate_detector.pt")
reader = easyocr.Reader(['en'])

# -----------------------------
# CLEAN FUNCTION
# -----------------------------
def clean_plate(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Fix common OCR errors
    replacements = {
        "O": "0",
        "C": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    fixed = ""
    for ch in text:
        fixed += replacements.get(ch, ch)

    return fixed

# -----------------------------
# SAVE TO CSV
# -----------------------------
def save_to_csv(plate, source):
    file_exists = os.path.isfile("plates.csv")

    with open("plates.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["PLATE", "SOURCE", "TIME"])

        writer.writerow([plate, source, time.strftime("%Y-%m-%d %H:%M:%S")])

# -----------------------------
# PROCESS SINGLE FRAME
# -----------------------------
def process_frame(frame, source="image"):
    results = model(frame)
    detected_plates = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        plate = frame[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        # Improve OCR
        plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        plate = cv2.GaussianBlur(plate, (3, 3), 0)

        text_list = reader.readtext(plate, detail=0, paragraph=True)

        for text in text_list:
            raw_text = text.upper()

            junk = ["NUMBER", "PLATE", "VEHICLE", "CAR", "BOX"]
            if any(j in raw_text for j in junk):
                continue

            clean_text = clean_plate(raw_text)

            if 8 <= len(clean_text) <= 12:
                print("\nRAW:", raw_text)
                print("PLATE:", clean_text)

                detected_plates.append(clean_text)
                save_to_csv(clean_text, source)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, detected_plates

# -----------------------------
# MODE 1: IMAGE
# -----------------------------
def run_image(path):
    image = cv2.imread(path)

    if image is None:
        print("Image not found:", path)
        return

    frame, plates = process_frame(image, path)

    cv2.imshow("ANPR - IMAGE", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# MODE 2: FOLDER BATCH
# -----------------------------
def run_folder(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if not path.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        print("\nProcessing:", path)
        image = cv2.imread(path)

        if image is None:
            continue

        frame, plates = process_frame(image, file)

        cv2.imshow("ANPR - BATCH", frame)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

# -----------------------------
# MODE 3: WEBCAM (REAL TIME)
# -----------------------------
def run_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame, plates = process_frame(frame, "webcam")

        cv2.imshow("ANPR - LIVE", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# MAIN CONTROLLER
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("""
Usage:
  python auto_anpr.py image.jpg
  python auto_anpr.py folder/
  python auto_anpr.py webcam
        """)
        exit()

    input_path = sys.argv[1]

    if input_path == "webcam":
        run_webcam()

    elif os.path.isdir(input_path):
        run_folder(input_path)

    else:
        run_image(input_path)