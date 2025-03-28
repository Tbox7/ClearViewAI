import cv2
import os
import face_recognition
import numpy as np
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from msrest.authentication import CognitiveServicesCredentials

__all__ = ["AzureVision", "AzureSpeech", "FaceRecognizer"]

class AzureVision:
    def __init__(self, key, endpoint):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def extract_text(self, frame):
        _, img_bytes = cv2.imencode(".jpg", frame)
        img_stream = BytesIO(img_bytes.tobytes())
        result = self.client.read_in_stream(img_stream, raw=True)
        operation = result.headers["Operation-Location"].split("/")[-1]
        while True:
            read_result = self.client.get_read_result(operation)
            if read_result.status.lower() not in ["notstarted", "running"]:
                break
        if read_result.status.lower() == "succeeded":
            return " ".join([line.text for r in read_result.analyze_result.read_results for line in r.lines])
        return None

    def describe_image(self, frame):
        _, img_bytes = cv2.imencode(".jpg", frame)
        img_stream = BytesIO(img_bytes.tobytes())
        result = self.client.analyze_image_in_stream(img_stream, visual_features=[VisualFeatureTypes.objects])
        if result.objects:
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            centered_objects = sorted(
                result.objects,
                key=lambda obj: (abs((obj.rectangle.x + obj.rectangle.w / 2) - frame_center_x) +
                                 abs((obj.rectangle.y + obj.rectangle.h / 2) - frame_center_y))
            )
            front_object = centered_objects[0]
            return f"In front of you is a {front_object.object_property}"
        return "No object detected."

class AzureSpeech:
    def __init__(self, key, region):
        self.config = SpeechConfig(subscription=key, region=region)
        self.synthesizer = SpeechSynthesizer(
            speech_config=self.config,
            audio_config=AudioOutputConfig(use_default_speaker=True)
        )

    def speak(self, text):
        print(f"[TTS] {text}")
        self.synthesizer.speak_text_async(text)

class FaceRecognizer:
    def __init__(self, faces_dir="faces"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(faces_dir)

    def load_known_faces(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".png") or file.endswith(".jpg"):
                name = os.path.splitext(file)[0].replace("_", " ").strip()
                image = face_recognition.load_image_file(os.path.join(directory, file))
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    self.known_face_names.append(name)

    def recognize_face(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1].astype(np.uint8)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if not face_locations:
            return "No face detected"
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        if not face_encodings:
            return "Face detected, but encoding failed"
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            return name
        return "Unknown person"
