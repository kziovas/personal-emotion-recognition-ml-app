import os
import pickle
from dataclasses import dataclass, field

import cv2
import numpy as np
from django.conf import settings

STATIC_DIR = settings.STATIC_DIR


# face detection model path
face_detector_model = cv2.dnn.readNetFromCaffe(
    os.path.join(STATIC_DIR, "models/deploy.prototxt.txt"),
    os.path.join(STATIC_DIR, "models/res10_300x300_ssd_iter_140000.caffemodel"),
)
# feature  model path
face_feature_model = cv2.dnn.readNetFromTorch(
    os.path.join(STATIC_DIR, "models/openface.nn4.small2.v1.t7")
)
# emotion recognition model path
emotion_recognition_model = pickle.load(
    open(os.path.join(STATIC_DIR, "models/machinelearning_face_emotion.pkl"), mode="rb")
)


@dataclass
class DetectedFaceResults:
    face_count: int
    face_confidence_score: float
    emotion_class: str
    emotion_confidence: float


@dataclass
class ImageResults:
    detected_faces: list[DetectedFaceResults] = field(default_factory=list)


class EmotionRecognitionMLPipeline:
    @classmethod
    def analyse_image(cls, image_path: str):
        # load and format image
        img = cv2.imread(image_path)
        image = img.copy()
        h, w = img.shape[:2]
        # face detection
        img_blob = cv2.dnn.blobFromImage(
            img, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False
        )
        face_detector_model.setInput(img_blob)
        detections = face_detector_model.forward()

        # machcine learning results
        image_results = ImageResults()
        count = 1
        if len(detections) > 0:
            for i, confidence in enumerate(detections[0, 0, :, 2]):
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    startx, starty, endx, endy = box.astype(int)

                    cv2.rectangle(image, (startx, starty), (endx, endy), (0, 255, 0))

                    # feature extraction
                    face_roi = img[starty:endy, startx:endx]
                    face_image_blob = cv2.dnn.blobFromImage(
                        face_roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True
                    )
                    face_feature_model.setInput(face_image_blob)
                    feature_vectors = face_feature_model.forward()

                    # emotion recognition
                    emotion_class = emotion_recognition_model.predict(feature_vectors)[
                        0
                    ]
                    emotion_confidence = emotion_recognition_model.predict_proba(
                        feature_vectors
                    ).max()

                    emotion_text = f"{emotion_class}: {emotion_confidence * 100:.0f}%"
                    cv2.putText(
                        image,
                        emotion_text,
                        (startx, endy),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imwrite(
                        os.path.join(settings.MEDIA_ROOT, "ml_output/analysed.jpg"),
                        image,
                    )
                    cv2.imwrite(
                        os.path.join(
                            settings.MEDIA_ROOT, f"ml_output/detection_{count}.jpg"
                        ),
                        face_roi,
                    )

                    face_results = DetectedFaceResults(
                        face_count=count,
                        face_confidence_score=confidence,
                        emotion_class=emotion_class,
                        emotion_confidence=emotion_confidence,
                    )
                    image_results.detected_faces.append(face_results)

                    count += 1

        return image_results
