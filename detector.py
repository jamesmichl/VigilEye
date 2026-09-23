import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import config


class DrowsinessDetector:

    # MediaPipe landmark indexes

    LEFT_EYE = [
        33,
        160,
        158,
        133,
        153,
        144
    ]

    RIGHT_EYE = [
        362,
        385,
        387,
        263,
        373,
        380
    ]

    # Mouth landmarks:
    # left corner, upper lip, right corner, lower lip
    MOUTH = [
        61,
        13,
        291,
        14
    ]

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

        # Calibration
        self.baseline_ear = None
        self.calibration_samples = []
        self.calibration_start = time.monotonic()

        # Eye closure
        self.eye_closed_start = None

        # Yawn
        self.yawn_start = None
        self.yawn_detected = False
        self.yawn_count = 0
        self.current_yawn_counted = False

        # PERCLOS history:
        # (timestamp, eyes_closed)
        self.eye_history = deque()

        self.status = "CALIBRATING"

    # Geometry

    @staticmethod
    def distance(point_a, point_b):
        return np.linalg.norm(
            np.array(point_a, dtype=np.float32)
            - np.array(point_b, dtype=np.float32)
        )

    @staticmethod
    def landmark_to_point(landmark, width, height):
        return (
            int(landmark.x * width),
            int(landmark.y * height)
        )

    def get_points(
        self,
        landmarks,
        indexes,
        width,
        height
    ):
        return [
            self.landmark_to_point(
                landmarks[index],
                width,
                height
            )
            for index in indexes
        ]

    # EAR

    def calculate_ear(self, eye):

        vertical_1 = self.distance(
            eye[1],
            eye[5]
        )

        vertical_2 = self.distance(
            eye[2],
            eye[4]
        )

        horizontal = self.distance(
            eye[0],
            eye[3]
        )

        if horizontal <= 1e-6:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (
            2.0 * horizontal
        )

    # MAR

    def calculate_mar(self, mouth):

        horizontal = self.distance(
            mouth[0],
            mouth[2]
        )

        vertical = self.distance(
            mouth[1],
            mouth[3]
        )

        if horizontal <= 1e-6:
            return 0.0

        return vertical / horizontal

    # Calibration

    def calibrate(self, ear):

        elapsed = (
            time.monotonic()
            - self.calibration_start
        )

        # Ignore first second while landmarks stabilize
        if elapsed >= config.CALIBRATION_WARMUP:
            self.calibration_samples.append(ear)

        if elapsed >= config.CALIBRATION_DURATION:

            if len(self.calibration_samples) < 10:
                self.reset_calibration()
                return False

            samples = np.array(
                self.calibration_samples,
                dtype=np.float32
            )

            # Remove obvious invalid values
            samples = samples[
                np.isfinite(samples)
                & (samples > 0)
            ]

            if len(samples) < 10:
                self.reset_calibration()
                return False

            # Remove extreme outliers using percentiles.
            # This also reduces the influence of blinks.
            lower = np.percentile(samples, 20)
            upper = np.percentile(samples, 95)

            filtered = samples[
                (samples >= lower)
                & (samples <= upper)
            ]

            if len(filtered) < 5:
                filtered = samples

            self.baseline_ear = float(
                np.median(filtered)
            )

            print("\nCalibration complete")
            print(
                f"Baseline EAR: "
                f"{self.baseline_ear:.3f}"
            )

            print(
                f"Closed-eye threshold: "
                f"{self.get_ear_threshold():.3f}"
            )

            return True

        return False

    def reset_calibration(self):

        self.baseline_ear = None
        self.calibration_samples = []
        self.calibration_start = time.monotonic()

        self.eye_closed_start = None

        self.eye_history.clear()

        self.yawn_start = None
        self.yawn_detected = False
        self.current_yawn_counted = False

        self.status = "CALIBRATING"

        print("\nCalibration restarted.")

    def get_ear_threshold(self):

        if self.baseline_ear is None:
            return None

        return (
            self.baseline_ear
            * config.EAR_THRESHOLD_RATIO
        )

    # PERCLOS

    def update_perclos(
        self,
        timestamp,
        eyes_closed
    ):

        self.eye_history.append(
            (timestamp, eyes_closed)
        )

        cutoff = (
            timestamp
            - config.PERCLOS_WINDOW_SECONDS
        )

        while (
            self.eye_history
            and self.eye_history[0][0] < cutoff
        ):
            self.eye_history.popleft()

    def calculate_perclos(self):

        if not self.eye_history:
            return 0.0

        closed_frames = sum(
            1
            for _, closed in self.eye_history
            if closed
        )

        return (
            closed_frames
            / len(self.eye_history)
        )

    # Yawn
    def update_yawn(self, mar, timestamp):

        if mar >= config.MAR_THRESHOLD:

            if self.yawn_start is None:
                self.yawn_start = timestamp

            duration = (
                timestamp
                - self.yawn_start
            )

            if (
                duration
                >= config.YAWN_MIN_DURATION
            ):

                self.yawn_detected = True

                if not self.current_yawn_counted:
                    self.yawn_count += 1
                    self.current_yawn_counted = True

        else:

            self.yawn_start = None
            self.yawn_detected = False
            self.current_yawn_counted = False

    # Drowsiness score

    def calculate_drowsiness_score(
        self,
        closed_duration,
        perclos,
        yawn
    ):

        score = 0

        # Prolonged eye closure
        if closed_duration >= config.WARNING_DURATION:
            score += 2

        if closed_duration >= config.DROWSY_DURATION:
            score += 2

        # PERCLOS
        if perclos >= config.PERCLOS_WARNING_THRESHOLD:
            score += 1

        if perclos >= config.PERCLOS_DROWSY_THRESHOLD:
            score += 1

        # Yawning
        if yawn:
            score += 1

        return score

    # Process frame

    def process(self, frame):

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        data = {
            "face_detected": False,
            "calibrated": (
                self.baseline_ear is not None
            ),
            "ear": 0.0,
            "relative_ear": 0.0,
            "threshold": 0.0,
            "mar": 0.0,
            "perclos": 0.0,
            "closed_duration": 0.0,
            "yawn": False,
            "yawn_count": self.yawn_count,
            "score": 0,
            "status": self.status,
            "left_eye": [],
            "right_eye": [],
            "mouth": [],
            "calibration_progress": 0.0
        }

        if not results.multi_face_landmarks:

            self.eye_closed_start = None

            return data

        data["face_detected"] = True

        face = results.multi_face_landmarks[0]
        landmarks = face.landmark

        # Eye points
        left_eye = self.get_points(
            landmarks,
            self.LEFT_EYE,
            width,
            height
        )

        right_eye = self.get_points(
            landmarks,
            self.RIGHT_EYE,
            width,
            height
        )

        # Mouth
        mouth = self.get_points(
            landmarks,
            self.MOUTH,
            width,
            height
        )

        data["left_eye"] = left_eye
        data["right_eye"] = right_eye
        data["mouth"] = mouth

        left_ear = self.calculate_ear(
            left_eye
        )

        right_ear = self.calculate_ear(
            right_eye
        )

        ear = (
            left_ear + right_ear
        ) / 2.0

        mar = self.calculate_mar(
            mouth
        )

        data["ear"] = ear
        data["mar"] = mar

        # Calibration

        if self.baseline_ear is None:

            elapsed = (
                time.monotonic()
                - self.calibration_start
            )

            progress = min(
                elapsed
                / config.CALIBRATION_DURATION,
                1.0
            )

            data["calibration_progress"] = progress

            self.status = "CALIBRATING"
            data["status"] = self.status

            self.calibrate(ear)

            data["calibrated"] = (
                self.baseline_ear is not None
            )

            return data

        # Detection
        timestamp = time.monotonic()

        threshold = self.get_ear_threshold()

        relative_ear = (
            ear / self.baseline_ear
            if self.baseline_ear > 0
            else 0
        )

        eyes_closed = (
            ear < threshold
        )

        # Eye closure duration
        if eyes_closed:

            if self.eye_closed_start is None:
                self.eye_closed_start = timestamp

            closed_duration = (
                timestamp
                - self.eye_closed_start
            )

        else:

            self.eye_closed_start = None
            closed_duration = 0.0

        # PERCLOS
        self.update_perclos(
            timestamp,
            eyes_closed
        )

        perclos = self.calculate_perclos()

        # Yawn
        self.update_yawn(
            mar,
            timestamp
        )

        # Score
        score = self.calculate_drowsiness_score(
            closed_duration,
            perclos,
            self.yawn_detected
        )

        # Status
        if score >= config.DROWSY_SCORE:
            self.status = "DROWSY"

        elif score >= config.WARNING_SCORE:
            self.status = "WARNING"

        else:
            self.status = "NORMAL"

        data.update({
            "calibrated": True,
            "ear": ear,
            "relative_ear": relative_ear,
            "threshold": threshold,
            "mar": mar,
            "perclos": perclos,
            "closed_duration": closed_duration,
            "yawn": self.yawn_detected,
            "yawn_count": self.yawn_count,
            "score": score,
            "status": self.status
        })

        return data
    
    # Cleanup

    def close(self):
        self.face_mesh.close()