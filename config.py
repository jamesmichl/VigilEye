# Camera
CAMERA_INDEX = 0

# Calibration
CALIBRATION_DURATION = 5.0
CALIBRATION_WARMUP = 1.0

# Adaptive Eye Aspect Ratio
# Eye is considered closed when current EAR falls below
# this percentage of the user's calibrated baseline.
EAR_THRESHOLD_RATIO = 0.70

# Duration thresholds (seconds)
WARNING_DURATION = 0.8
DROWSY_DURATION = 1.5

# PERCLOS
# Percentage of recent frames where the eyes were closed.
PERCLOS_WINDOW_SECONDS = 20
PERCLOS_WARNING_THRESHOLD = 0.30
PERCLOS_DROWSY_THRESHOLD = 0.45

# Yawn detection
MAR_THRESHOLD = 0.60
YAWN_MIN_DURATION = 1.0

# Drowsiness score
WARNING_SCORE = 2
DROWSY_SCORE = 4

# MediaPipe
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# UI
WINDOW_NAME = "Drowsiness Detection System"