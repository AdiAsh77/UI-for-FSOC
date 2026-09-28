import cv2
import numpy as np
import csv
import math
import time
import os
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "input.mp4"
OUTPUT_VIDEO_PATH = "tracked_output.mp4"
OUTPUT_CSV_PATH = "tracking_data.csv"

MODEL_PATH = "newbest.pt"

CONFIDENCE_THRESHOLD = 0.35

# Run YOLO every N frames after acquisition
YOLO_INTERVAL = 3

# Number of detections required for acquisition
INITIALIZATION_DETECTIONS = 20

# Tracking parameters
TRACK_CAMERA_FPS = 30.0

# RMSE window
RMSE_WINDOW = 300

# Accuracy threshold
ACCURACY_THRESHOLD = 12.0

# Lock threshold
LOCK_ERROR_THRESHOLD = 60.0

# ============================================================
# SIMULATED PAN / TILT
# ============================================================

# Gain:
# Converts pixel error into angular velocity.
#
# Example:
# error_x = 20 pixels
# 20 * 0.05 = 1 degree/second
#
PAN_GAIN = 0.05
TILT_GAIN = 0.05

# Maximum angular speed in degrees/second
MAX_PAN_SPEED = 2.0
MAX_TILT_SPEED = 2.0


# ============================================================
# CSV OUTPUT
# ============================================================

CSV_HEADER = [
    "time",
    "frame",
    "offset_x",
    "offset_y",
    "error",
    "pan",
    "tilt",
    "rmse",
    "lock_retention",
    "accuracy",
    "target_losses",
    "confidence",
    "detection"
]


def csv_row(result):

    return [
        f"{result['time']:.4f}",
        result["frame_number"],
        f"{result['offset_x']:.4f}",
        f"{result['offset_y']:.4f}",
        f"{result['error']:.4f}",
        f"{result['pan']:.4f}",
        f"{result['tilt']:.4f}",
        f"{result['rmse']:.4f}",
        f"{result['lock_retention']:.4f}",
        f"{result['accuracy']:.4f}",
        result["target_loss_count"],
        (
            f"{result['confidence']:.4f}"
            if result["confidence"] is not None
            else ""
        ),
        int(result["target_detected"])
    ]


def write_csv(rows, path=None):

    path = path or OUTPUT_CSV_PATH

    with open(path, "w", newline="") as csv_file:

        csv_writer = csv.writer(
            csv_file
        )

        csv_writer.writerow(
            CSV_HEADER
        )

        csv_writer.writerows(
            rows
        )

    return os.path.abspath(
        path
    )


# ============================================================
# YOLO MODEL
# ============================================================

_MODEL_CACHE = {}


def load_model(path=None):

    path = path or MODEL_PATH

    if path not in _MODEL_CACHE:

        _MODEL_CACHE[path] = YOLO(
            path
        )

    return _MODEL_CACHE[path]


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_beacon(model, frame):

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    if len(results) == 0:
        return None, None, None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None, None, None

    # Select highest confidence detection
    best_index = int(
        torch_argmax(result.boxes.conf)
    )

    confidence = float(
        result.boxes.conf[best_index]
    )

    box = result.boxes.xyxy[
        best_index
    ].cpu().numpy()

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    return (
        (center_x, center_y),
        confidence,
        (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )
    )


def torch_argmax(tensor):

    return int(
        tensor.argmax().item()
    )


# ============================================================
# KALMAN FILTER
# ============================================================

def create_kalman():

    kalman = cv2.KalmanFilter(4, 2)

    # State:
    #
    # x
    # y
    # vx
    # vy

    kalman.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        dtype=np.float32
    )

    kalman.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ],
        dtype=np.float32
    )

    kalman.processNoiseCov = np.array(
        [
            [1e-2, 0,    0,     0],
            [0,    1e-2, 0,     0],
            [0,    0,    1e-1, 0],
            [0,    0,    0,     1e-1]
        ],
        dtype=np.float32
    )

    kalman.measurementNoiseCov = np.array(
        [
            [1.5, 0],
            [0, 1.5]
        ],
        dtype=np.float32
    )

    kalman.errorCovPost = np.eye(
        4,
        dtype=np.float32
    )

    return kalman


# ============================================================
# SIMULATED UNITY PAN / TILT
# ============================================================

def update_simulated_pan_tilt(
    current_pan,
    current_tilt,
    error_x,
    error_y,
    previous_error_x,
    previous_error_y
):

    # --------------------------------------------------------
    # Calculate how much the beacon moved in the image
    # --------------------------------------------------------

    delta_x = error_x - previous_error_x
    delta_y = error_y - previous_error_y

    # --------------------------------------------------------
    # Small movement dead zone
    # Prevent tiny YOLO/Kalman fluctuations from moving
    # the simulated gimbal.
    # --------------------------------------------------------

    if abs(delta_x) < 2:
        delta_x = 0.0

    if abs(delta_y) < 2:
        delta_y = 0.0

    # --------------------------------------------------------
    # Convert image movement into gimbal movement
    # --------------------------------------------------------

    pan_change = delta_x * PAN_GAIN
    tilt_change = -delta_y * TILT_GAIN

    # --------------------------------------------------------
    # Limit how much the gimbal can move per frame
    # --------------------------------------------------------

    pan_change = np.clip(
        pan_change,
        -MAX_PAN_SPEED,
        MAX_PAN_SPEED
    )

    tilt_change = np.clip(
        tilt_change,
        -MAX_TILT_SPEED,
        MAX_TILT_SPEED
    )

    # --------------------------------------------------------
    # Update accumulated gimbal orientation
    # --------------------------------------------------------

    current_pan += pan_change
    current_tilt += tilt_change

    # --------------------------------------------------------
    # Normalize angles
    # --------------------------------------------------------

    current_pan = (
        (current_pan + 180)
        % 360
    ) - 180

    current_tilt = (
        (current_tilt + 180)
        % 360
    ) - 180

    return (
        current_pan,
        current_tilt
    )

# ============================================================
# RMSE
# ============================================================

def calculate_rmse(errors):

    if len(errors) == 0:
        return 0.0

    recent_errors = errors[
        -RMSE_WINDOW:
    ]

    squared_errors = [
        error ** 2
        for error in recent_errors
    ]

    return math.sqrt(
        sum(squared_errors)
        /
        len(squared_errors)
    )


# ============================================================
# BEACON TRACKER
# ============================================================

class BeaconTracker:

    def __init__(
        self,
        width,
        height,
        fps,
        model=None,
        verbose=True
    ):

        self.model = (
            model
            if model is not None
            else load_model()
        )

        self.width = width
        self.height = height

        self.fps = (
            fps
            if fps and fps > 0
            else TRACK_CAMERA_FPS
        )

        self.verbose = verbose

        # Image center
        self.center_x = width / 2.0
        self.center_y = height / 2.0

        self.reset()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.previous_error_x = None
        self.previous_error_y = None

        self.kalman = create_kalman()
        self.kalman_initialized = False

        # Acquisition
        self.acquisition_complete = False
        self.acquisition_count = 0
        self.acquisition_start_frame = None
        self.acquisition_end_frame = None
        self.acquisition_start_time = None
        self.acquisition_time = None
        self.acquisition_positions = []
        self.acquisition_times = []

        self.last_x = None
        self.last_y = None

        # Tracking
        self.current_pan = 0.0
        self.current_tilt = 0.0

        self.error_over_time = []

        self.target_loss_count = 0
        self.previous_detection = True

        self.accuracy_count = 0
        self.tracking_frame_count = 0

        self.frame_number = 0

        # Latest metrics
        self.rmse = 0.0
        self.lock_retention = 0.0
        self.accuracy = 0.0

        # Outcome of most recent YOLO run
        self.detected = False

        # Latest YOLO box size
        self.last_box_size = None

    # --------------------------------------------------------
    # PROCESS ONE FRAME
    # --------------------------------------------------------

    def process(self, frame):

        center_x = self.center_x
        center_y = self.center_y

        self.frame_number += 1

        frame_number = self.frame_number

        current_time = (
            frame_number / self.fps
        )

        # ----------------------------------------------------
        # DEFAULTS
        # ----------------------------------------------------

        target_detected = False

        detection_position = None
        detection_confidence = None
        detection_box = None

        detection_fired = False

        filtered_x = None
        filtered_y = None

        # ====================================================
        # ACQUISITION
        # ====================================================

        if not self.acquisition_complete:

            # Run YOLO every frame during acquisition

            detection_position, \
            detection_confidence, \
            detection_box = detect_beacon(
                self.model,
                frame
            )

            if detection_position is not None:

                target_detected = True

                x, y = detection_position

                # --------------------------------------------
                # Store acquisition measurement
                # --------------------------------------------

                self.acquisition_positions.append(
                    (x, y)
                )

                self.acquisition_times.append(
                    current_time
                )

                self.acquisition_count += 1

                self.last_x = x
                self.last_y = y

                # --------------------------------------------
                # Start acquisition timer
                # --------------------------------------------

                if self.acquisition_count == 1:

                    self.acquisition_start_frame = (
                        frame_number
                    )

                    self.acquisition_start_time = (
                        current_time
                    )

                # --------------------------------------------
                # Acquisition progress
                # --------------------------------------------

                error_x = (
                    x - center_x
                )

                error_y = (
                    y - center_y
                )

                if self.verbose:

                    print(
                        f"\rAcquiring: "
                        f"{self.acquisition_count}/"
                        f"{INITIALIZATION_DETECTIONS}"
                        f" | Beacon: "
                        f"({x:.1f}, {y:.1f})"
                        f" | Error: "
                        f"({error_x:.1f}, "
                        f"{error_y:.1f})"
                        f" | Conf: "
                        f"{detection_confidence:.2f}",
                        end=""
                    )

                # --------------------------------------------
                # Acquisition complete
                # --------------------------------------------

                if (
                    self.acquisition_count
                    >= INITIALIZATION_DETECTIONS
                ):

                    self.acquisition_complete = True

                    self.acquisition_end_frame = (
                        frame_number
                    )

                    self.acquisition_time = (
                        current_time
                        -
                        self.acquisition_start_time
                    )

                    # ----------------------------------------
                    # Initial velocity
                    # ----------------------------------------

                    if len(
                        self.acquisition_positions
                    ) >= 2:

                        previous_x, previous_y = (
                            self.acquisition_positions[-2]
                        )

                        previous_time = (
                            self.acquisition_times[-2]
                        )

                        dt = (
                            current_time
                            -
                            previous_time
                        )

                        if dt > 0:

                            vx = (
                                x - previous_x
                            ) / dt

                            vy = (
                                y - previous_y
                            ) / dt

                        else:

                            vx = 0.0
                            vy = 0.0

                    else:

                        vx = 0.0
                        vy = 0.0

                    # ----------------------------------------
                    # Initialize Kalman state
                    # ----------------------------------------

                    self.kalman.statePost = np.array(
                        [
                            [x],
                            [y],
                            [vx],
                            [vy]
                        ],
                        dtype=np.float32
                    )

                    self.kalman.statePre = np.array(
                        [
                            [x],
                            [y],
                            [vx],
                            [vy]
                        ],
                        dtype=np.float32
                    )

                    self.kalman_initialized = True

                    # ----------------------------------------
                    # Use acquisition position
                    # ----------------------------------------

                    filtered_x = self.last_x
                    filtered_y = self.last_y

                    if self.verbose:

                        print()
                        print()

                        print(
                            "================================"
                        )

                        print(
                            "ACQUISITION COMPLETE"
                        )

                        print(
                            "================================"
                        )

                        print(
                            f"Acquisition time: "
                            f"{self.acquisition_time:.3f} s"
                        )

                        print(
                            f"Acquisition frame: "
                            f"{self.acquisition_end_frame}"
                        )

                        print(
                            f"Initial position: "
                            f"({self.last_x:.1f}, "
                            f"{self.last_y:.1f})"
                        )

                        print()

            self.detected = target_detected

            # -----------------------------------------------
            # Position on acquisition completion frame
            # -----------------------------------------------

            if (
                self.acquisition_complete
                and
                filtered_x is None
            ):

                filtered_x = self.last_x
                filtered_y = self.last_y

        # ====================================================
        # TRACKING
        # ====================================================

        else:

            # --------------------------------------------
            # Determine whether YOLO should run
            # --------------------------------------------

            detection_fired = (
                (
                    frame_number
                    -
                    self.acquisition_end_frame
                )
                %
                YOLO_INTERVAL
                == 0
            )

            # --------------------------------------------
            # Kalman prediction
            # --------------------------------------------

            if self.kalman_initialized:

                prediction = (
                    self.kalman.predict()
                )

                filtered_x = float(
                    prediction[0, 0]
                )

                filtered_y = float(
                    prediction[1, 0]
                )

            # --------------------------------------------
            # YOLO correction
            # --------------------------------------------

            if detection_fired:

                detection_position, \
                detection_confidence, \
                detection_box = detect_beacon(
                    self.model,
                    frame
                )

                if detection_position is not None:

                    target_detected = True

                    detection_x, detection_y = (
                        detection_position
                    )

                    measurement = np.array(
                        [
                            [detection_x],
                            [detection_y]
                        ],
                        dtype=np.float32
                    )

                    corrected = (
                        self.kalman.correct(
                            measurement
                        )
                    )

                    filtered_x = float(
                        corrected[0, 0]
                    )

                    filtered_y = float(
                        corrected[1, 0]
                    )

                else:

                    target_detected = False

            # --------------------------------------------
            # Target loss
            # --------------------------------------------

            if detection_fired:

                if (
                    not target_detected
                    and
                    self.previous_detection
                ):

                    self.target_loss_count += 1

                self.previous_detection = (
                    target_detected
                )

                self.detected = target_detected

            # --------------------------------------------
            # Safety fallback
            # --------------------------------------------

            if filtered_x is None:

                if self.last_x is not None:

                    filtered_x = self.last_x
                    filtered_y = self.last_y

                else:

                    filtered_x = center_x
                    filtered_y = center_y

            # --------------------------------------------
            # Save latest position
            # --------------------------------------------

            self.last_x = filtered_x
            self.last_y = filtered_y

        # ====================================================
        # CALCULATE ERROR
        # ====================================================

        if (
            filtered_x is not None
            and
            filtered_y is not None
        ):

            offset_x = (
                filtered_x
                -
                center_x
            )

            offset_y = (
                filtered_y
                -
                center_y
            )

            error = math.sqrt(
                offset_x ** 2
                +
                offset_y ** 2
            )

        else:

            offset_x = 0.0
            offset_y = 0.0
            error = 0.0

        # ====================================================
        # SIMULATED PAN / TILT
        # ====================================================

        if self.acquisition_complete:

            if (
                self.previous_error_x is None
                or self.previous_error_y is None
            ):

                # Initial acquisition correction
                self.current_pan = np.clip(
                    offset_x * PAN_GAIN,
                    -180,
                    180
                )

                self.current_tilt = np.clip(
                    -offset_y * TILT_GAIN,
                    -180,
                    180
                )

                self.previous_error_x = offset_x
                self.previous_error_y = offset_y

            else:

                self.current_pan, self.current_tilt = (
                    update_simulated_pan_tilt(
                        self.current_pan,
                        self.current_tilt,
                        offset_x,
                        offset_y,
                        self.previous_error_x,
                        self.previous_error_y
                    )
                )

                self.previous_error_x = offset_x
                self.previous_error_y = offset_y

        # ====================================================
        # METRICS
        # ====================================================

        if self.acquisition_complete:

            self.tracking_frame_count += 1

            self.error_over_time.append(
                error
            )

            if error <= ACCURACY_THRESHOLD:

                self.accuracy_count += 1

            self.accuracy = (
                self.accuracy_count
                /
                self.tracking_frame_count
                *
                100.0
            )

        else:

            self.accuracy = 0.0

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        self.rmse = calculate_rmse(
            self.error_over_time
        )

        # ----------------------------------------------------
        # Lock retention
        # ----------------------------------------------------

        if self.acquisition_complete:

            locked_frames = sum(
                1
                for e in self.error_over_time
                if e < LOCK_ERROR_THRESHOLD
            )

            if len(self.error_over_time) > 0:

                self.lock_retention = (
                    locked_frames
                    /
                    len(self.error_over_time)
                    *
                    100.0
                )

            else:

                self.lock_retention = 0.0

        else:

            self.lock_retention = 0.0

        # ----------------------------------------------------
        # Current values
        # ----------------------------------------------------

        rmse = self.rmse
        accuracy = self.accuracy
        lock_retention = self.lock_retention

        current_pan = self.current_pan
        current_tilt = self.current_tilt

        target_loss_count = (
            self.target_loss_count
        )

        acquisition_complete = (
            self.acquisition_complete
        )

        # ====================================================
        # DRAW ANNOTATIONS
        # ====================================================

        # ----------------------------------------------------
        # Image center
        # ----------------------------------------------------

        cv2.drawMarker(
            frame,
            (
                int(center_x),
                int(center_y)
            ),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )

        # ----------------------------------------------------
        # YOLO bounding box
        # ----------------------------------------------------

        if detection_box is not None:

            x1, y1, x2, y2 = (
                detection_box
            )

            self.last_box_size = (
                x2 - x1,
                y2 - y1
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            if detection_position is not None:

                dx, dy = (
                    detection_position
                )

                cv2.circle(
                    frame,
                    (
                        int(dx),
                        int(dy)
                    ),
                    5,
                    (0, 255, 0),
                    -1
                )

        # ----------------------------------------------------
        # Kalman position
        # ----------------------------------------------------

        if (
            acquisition_complete
            and
            filtered_x is not None
            and
            filtered_y is not None
        ):

            box_w, box_h = (
                self.last_box_size
                if self.last_box_size is not None
                else (30, 30)
            )

            half_w = (
                box_w / 2.0
                +
                4
            )

            half_h = (
                box_h / 2.0
                +
                4
            )

            cv2.rectangle(
                frame,
                (
                    int(filtered_x - half_w),
                    int(filtered_y - half_h)
                ),
                (
                    int(filtered_x + half_w),
                    int(filtered_y + half_h)
                ),
                (255, 0, 0),
                2
            )

            cv2.line(
                frame,
                (
                    int(center_x),
                    int(center_y)
                ),
                (
                    int(filtered_x),
                    int(filtered_y)
                ),
                (255, 0, 0),
                2
            )

        # ====================================================
        # TEXT
        # ====================================================

        status = (
            "ACQUIRING"
            if not acquisition_complete
            else "TRACKING"
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Mode: {status}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Offset: "
            f"({offset_x:.1f}, "
            f"{offset_y:.1f})",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Error: {error:.1f}px",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"RMSE: {rmse:.2f}px",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Pan: {current_pan:.2f}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tilt: {current_tilt:.2f}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Lock: {lock_retention:.1f}%",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Accuracy: {accuracy:.1f}%",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Target Losses: "
            f"{target_loss_count}",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Detection status
        # ----------------------------------------------------

        if (
            detection_fired
            if acquisition_complete
            else True
        ):

            if target_detected:

                detection_text = (
                    "YOLO: DETECTED"
                )

            else:

                detection_text = (
                    "YOLO: LOST"
                )

        else:

            detection_text = (
                "YOLO: SKIPPED"
            )

        cv2.putText(
            frame,
            detection_text,
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # RESULT
        # ====================================================

        if detection_position is not None:

            beacon_offset = (
                detection_position[0]
                -
                center_x,
                detection_position[1]
                -
                center_y
            )

        elif acquisition_complete:

            beacon_offset = (
                offset_x,
                offset_y
            )

        else:

            beacon_offset = None

        return {
            "frame": frame,
            "frame_number": frame_number,
            "time": current_time,
            "status": status,
            "acquisition_complete":
                acquisition_complete,
            "target_detected":
                target_detected,
            "detection_fired":
                detection_fired,
            "detected":
                self.detected,
            "confidence":
                detection_confidence,
            "beacon_offset":
                beacon_offset,
            "offset_x":
                offset_x,
            "offset_y":
                offset_y,
            "error":
                error,
            "pan":
                current_pan,
            "tilt":
                current_tilt,
            "rmse":
                rmse,
            "lock_retention":
                lock_retention,
            "accuracy":
                accuracy,
            "target_loss_count":
                target_loss_count,
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("VIDEO TRACKING")
    print("==============================")

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            f"ERROR: Could not open video: "
            f"{VIDEO_PATH}"
        )

        return

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        fps = TRACK_CAMERA_FPS

    print(
        f"Video: {width}x{height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print(
            "ERROR: Could not create output video."
        )

        cap.release()

        return

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = load_model()

    print("YOLO model loaded.")
    print()

    # --------------------------------------------------------
    # TRACKER
    # --------------------------------------------------------

    tracker = BeaconTracker(
        width,
        height,
        fps,
        model=model
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_file = open(
        OUTPUT_CSV_PATH,
        "w",
        newline=""
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        CSV_HEADER
    )

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    frame_number = 0

    processing_start = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = tracker.process(
            frame
        )

        frame_number = (
            result["frame_number"]
        )

        # ====================================================
        # CSV
        # ====================================================

        csv_writer.writerow(
            csv_row(result)
        )

        # ====================================================
        # WRITE OUTPUT VIDEO
        # ====================================================

        writer.write(
            frame
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    processing_time = (
        time.time()
        -
        processing_start
    )

    cap.release()

    writer.release()

    csv_file.close()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    acquisition_complete = (
        tracker.acquisition_complete
    )

    print()
    print()

    print("==============================")
    print("VIDEO TRACKING COMPLETE")
    print("==============================")

    print(
        f"Frames processed: "
        f"{frame_number}"
    )

    print(
        f"Processing time: "
        f"{processing_time:.2f} s"
    )

    if frame_number > 0:

        print(
            f"Processing FPS: "
            f"{frame_number / processing_time:.2f}"
        )

    if acquisition_complete:

        print()
        print("ACQUISITION")

        print(
            f"Acquisition time: "
            f"{tracker.acquisition_time:.3f} s"
        )

        print(
            f"Acquisition frame: "
            f"{tracker.acquisition_end_frame}"
        )

        print()
        print("TRACKING")

        print(
            f"Tracking frames: "
            f"{tracker.tracking_frame_count}"
        )

        print(
            f"RMSE: "
            f"{tracker.rmse:.2f} px"
        )

        print(
            f"Lock retention: "
            f"{tracker.lock_retention:.2f}%"
        )

        print(
            f"Accuracy <= "
            f"{ACCURACY_THRESHOLD:.1f}px: "
            f"{tracker.accuracy:.2f}%"
        )

        print(
            f"Target losses: "
            f"{tracker.target_loss_count}"
        )

        print()
        print("FINAL SIMULATED PAN/TILT")

        print(
            f"Pan: "
            f"{tracker.current_pan:.2f}°"
        )

        print(
            f"Tilt: "
            f"{tracker.current_tilt:.2f}°"
        )

    else:

        print()
        print(
            "WARNING: Acquisition "
            "did not complete."
        )

    print()

    print(
        f"Output video: "
        f"{OUTPUT_VIDEO_PATH}"
    )

    print(
        f"Output CSV: "
        f"{OUTPUT_CSV_PATH}"
    )

    print(
        "=============================="
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()