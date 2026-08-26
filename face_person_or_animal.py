"""
Controls:
    Press 'q' to quit the live window.

Requirements:
    pip install opencv-python
"""

import cv2

PERSON_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ANIMAL_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml"

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (60, 60)

PERSON_COLOR = (0, 255, 0)   # green
ANIMAL_COLOR = (255, 140, 0)  # orange


def boxes_overlap(box_a, box_b):
    """Check if two (x, y, w, h) boxes overlap significantly, to avoid double-labeling the same face."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller_area = min(aw * ah, bw * bh)
    return intersection / smaller_area > 0.4


def run_detection(camera_index=0):
    person_cascade = cv2.CascadeClassifier(PERSON_CASCADE_PATH)
    animal_cascade = cv2.CascadeClassifier(ANIMAL_CASCADE_PATH)

    if person_cascade.empty() or animal_cascade.empty():
        raise IOError("Failed to load one or more Haar cascade files.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Could not open webcam. Try a different camera_index (0, 1, 2...).")

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        person_faces = person_cascade.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE
        )
        animal_faces = animal_cascade.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE
        )

        # Drop animal detections that overlap a person detection (avoids double labels
        # on the same region, since cascades can both fire on similar patterns).
        filtered_animal_faces = [
            a for a in animal_faces
            if not any(boxes_overlap(a, p) for p in person_faces)
        ]

        for (x, y, w, h) in person_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), PERSON_COLOR, 2)
            cv2.putText(frame, "Person", (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)

        for (x, y, w, h) in filtered_animal_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), ANIMAL_COLOR, 2)
            cv2.putText(frame, "Animal", (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANIMAL_COLOR, 2)

        total = len(person_faces) + len(filtered_animal_faces)
        summary = f"Person: {len(person_faces)}  Animal: {len(filtered_animal_faces)}  Total: {total}"
        cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Face Detection - Person vs Animal (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection(camera_index=0)

