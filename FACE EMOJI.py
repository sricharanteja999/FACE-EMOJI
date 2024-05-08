import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self, face_cascade_path, model_path, emotion_labels):
        self.face_classifier = cv2.CascadeClassifier(face_cascade_path)
        self.classifier = load_model(model_path)
        self.emotion_labels = emotion_labels

    def detect_emotions(self, frame):
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.classifier.predict(roi)[0]
                label = self.emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def main():
    face_cascade_path = r'C:\Users\SRICHARANTEJA\OneDrive\Desktop\face\haarcascade_frontalface_default.xml'
    model_path = r'C:\Users\SRICHARANTEJA\OneDrive\Desktop\face\model.h5'
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)
    detector = EmotionDetector(face_cascade_path, model_path, emotion_labels)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect_emotions(frame)
        cv2.imshow('Emotion Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
