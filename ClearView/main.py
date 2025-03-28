import cv2
from assistant import AzureVision, AzureSpeech, FaceRecognizer

AZURE_VISION_KEY = ""
AZURE_VISION_ENDPOINT = ""
AZURE_SPEECH_KEY = ""
AZURE_SPEECH_REGION = ""

vision = AzureVision(AZURE_VISION_KEY, AZURE_VISION_ENDPOINT)
speech = AzureSpeech(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
face_recognizer = FaceRecognizer("faces")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    exit()

print("Press 'r' to read text, 'i' to identify object, 'f' to recognize face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame.")
        break

    cv2.imshow("Meta Glasses Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        text = vision.extract_text(frame)
        speech.speak(text if text else "No text detected.")

    elif key == ord('i'):
        description = vision.describe_image(frame)
        speech.speak(description)

    elif key == ord('f'):
        name = face_recognizer.recognize_face(frame)
        speech.speak(f"I see {name}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
