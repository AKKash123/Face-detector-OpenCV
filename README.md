#face_person_or_animal.py

#Detects faces in real time via webcam and classifies each one as:

#Person face   (Haar cascade: haarcascade_frontalface_default.xml)
#Animal face   (Haar cascade: haarcascade_frontalcatface_extended.xml)

Both cascades ship with opencv-python, so no downloads are needed.

Note on scope: OpenCV's built-in Haar cascades only include a robust
face detector for humans and cats. If you need broader animal-face
coverage (dogs, horses, etc.), a deep-learning model trained on animal
faces would be required instead — see the note at the bottom of this
file for how to extend this script that way.

Usage:
    python face_person_or_animal.py

Controls:
    Press 'q' to quit the live window.

Requirements:
    pip install opencv-python
