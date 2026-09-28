import socket
import struct
import threading
import time
import math
from collections import deque
from config import unity, sc

import numpy as np
import cv2
from ultralytics import YOLO

from PyQt6.QtCore import QThread, pyqtSignal

from parser import parse_scene_file
import json


# ============================================================
# CONFIGURATION
# ============================================================

TRACK_HOST = "127.0.0.1"
TRACK_PORT = 5000

MODEL_PATH = "newbest.pt"

CONFIDENCE_THRESHOLD = 0.35

YOLO_INTERVAL = 3

INITIALIZATION_DETECTIONS = 20

ACQUISITION_ERROR_X = 40.0
ACQUISITION_ERROR_Y = 40.0

TRACK_CAMERA_FPS = 30.0


# Professor's RMSE definition:
# RMSE of hypot(ox, oy) over the last 300 frames

RMSE_WINDOW = 300


# Professor's lock definition

LOCK_ERROR_THRESHOLD = 60.0


# Accuracy definition

ACCURACY_THRESHOLD = 12.0


# ============================================================
# SEARCH CONFIGURATION
# ============================================================

BEACON_LOST_TIMEOUT = 10.0

SEARCH_CONFIRMATIONS_REQUIRED = 3


UNITY_EXE = unity


# ============================================================
# KALMAN FILTER
# ============================================================

class BeaconKalmanFilter:

    def __init__(self):

        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.measurementNoiseCov = np.array([
            [1.5, 0],
            [0, 1.5]
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.array([
            [1e-2, 0,    0,    0],
            [0,    1e-2, 0,    0],
            [0,    0,    1e-1, 0],
            [0,    0,    0,    1e-1]
        ], dtype=np.float32)

        self.initialized = False


    def predict(self, dt):

        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        prediction = self.kalman.predict()

        return (
            float(prediction[0, 0]),
            float(prediction[1, 0])
        )


    def update(self, x, y):

        measurement = np.array([
            [np.float32(x)],
            [np.float32(y)]
        ])

        corrected = self.kalman.correct(
            measurement
        )

        return (
            float(corrected[0, 0]),
            float(corrected[1, 0])
        )


    def initialize(self, x, y, vx, vy):

        self.kalman.statePost = np.array([
            [np.float32(x)],
            [np.float32(y)],
            [np.float32(vx)],
            [np.float32(vy)]
        ], dtype=np.float32)

        self.kalman.statePre = (
            self.kalman.statePost.copy()
        )

        self.kalman.errorCovPost = np.eye(
            4,
            dtype=np.float32
        )

        self.initialized = True


# ============================================================
# REAL TRACKING THREAD
# ============================================================

class TrackingThread(QThread):

    frame_ready = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str)
    stats_ready = pyqtSignal(dict)


    def __init__(self):

        super().__init__()

        self._running = True

        # ----------------------------------------------------
        # PAUSE STATE
        # ----------------------------------------------------

        self._paused = False
        self._pause_lock = threading.Lock()

        # ----------------------------------------------------
        # SCENE RESET
        # ----------------------------------------------------

        self._scene_reset_requested = threading.Event()

        # ----------------------------------------------------
        # SOCKETS
        # ----------------------------------------------------

        self.server = None
        self.conn = None

        # ----------------------------------------------------
        # PAN / TILT
        # ----------------------------------------------------

        self.current_pan = 0.0
        self.current_tilt = 0.0

        self.pan_lock = threading.Lock()


    # ========================================================
    # REQUEST TRACKING RESET
    # ========================================================

    def request_tracking_reset(self):

        print()
        print("==========================================")
        print("NEW SCENE DETECTED")
        print("REQUESTING COMPLETE TRACKING RESET")
        print("==========================================")

        self._scene_reset_requested.set()


    # ========================================================
    # PAUSE
    # ========================================================

    def pause(self):

        with self._pause_lock:
            self._paused = True

        if self.conn is not None:

            try:

                self.conn.sendall(
                    b"PAUSE\n"
                )

                print()
                print(
                    "PAUSE command sent to Unity."
                )

            except Exception as e:

                print(
                    "Pause command error:",
                    e
                )

        self.status_changed.emit(
            "Paused"
        )


    # ========================================================
    # RESUME
    # ========================================================

    def resume(self):

        if self.conn is not None:

            try:

                self.conn.sendall(
                    b"RESUME\n"
                )

                print()
                print(
                    "RESUME command sent to Unity."
                )

            except Exception as e:

                print(
                    "Resume command error:",
                    e
                )

        with self._pause_lock:
            self._paused = False

        self.status_changed.emit(
            "Resumed"
        )


    # ========================================================
    # CHECK PAUSED
    # ========================================================

    def is_paused(self):

        with self._pause_lock:

            return self._paused

    def type_chosen(self, val):
        print("movement type: ", val)

    # ========================================================
    # RESET PAN / TILT
    # ========================================================

    def reset_pan_tilt(self):

        with self.pan_lock:

            self.current_pan = 0.0
            self.current_tilt = 0.0


    # ========================================================
    # UNITY VIEW CHANGE
    # ========================================================

    def send_scene(self, scene):

        if self.conn is None:
            return

        scene_json = json.dumps(scene)

        message = scene_json + "\n"

        try:

            self.conn.sendall(
                bytes([3]) +
                message.encode("utf-8")
            )

            print()
            print(
                "NEW SCENE SENT TO UNITY."
            )

        except Exception as e:

            print(
                "Scene send error:",
                e
            )


    def parsed(self, val):

        print()
        print("==========================================")
        print(
            "LOADING NEW SCENE:",
            val
        )
        print("==========================================")

        st = f"{sc}{val}.txt"

        scene = parse_scene_file(
            st
        )

        print()
        print("SCENE DATA:")
        print(scene)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Stop the current tracking before applying
        # the new scene.
        # ----------------------------------------------------

        self.pause()

        # ----------------------------------------------------
        # Tell tracking thread to completely reset.
        # ----------------------------------------------------

        self.request_tracking_reset()

        # ----------------------------------------------------
        # Send the new scene to Unity.
        #
        # Unity can still receive this while paused because
        # CameraStreamer.ReceiveCoordinates() continues
        # running.
        # ----------------------------------------------------

        self.send_scene(scene)

        # ----------------------------------------------------
        # Start tracking again.
        #
        # The reset itself will be processed by the tracking
        # thread before the next frame is processed.
        # ----------------------------------------------------

        self.resume()

        print()
        print("NEW SCENE READY.")
        print("PYTHON TRACKING WILL START FROM SCRATCH.")
        print()


    # ========================================================
    # SATELLITE POV
    # ========================================================

    def sat_pov(self):

        with self._pause_lock:
            self._paused = False

        if self.conn is not None:

            try:

                self.conn.sendall(
                    b"SAT_POV\n"
                )

                print()
                print(
                    "SAT_POV command sent to Unity."
                )

            except Exception as e:

                print(
                    "SAT_POV command error:",
                    e
                )


    # ========================================================
    # PAN / TILT
    # ========================================================

    def placehold(self, value):

        pan = value[0]
        tilt = value[1]

        message = (
            f"PANTILT,"
            f"{pan:.4f},"
            f"{tilt:.4f}\n"
        )

        if self.conn is not None:

            self.conn.sendall(
                message.encode()
            )

        print(
            "dont worry sending: ",
            value
        )


    # ========================================================
    # SECOND VIEW
    # ========================================================

    def second_view(self):

        with self._pause_lock:
            self._paused = False

        if self.conn is not None:

            try:

                self.conn.sendall(
                    b"SECOND_VIEW\n"
                )

                print()
                print(
                    "SECOND_VIEW command sent to Unity."
                )

            except Exception as e:

                print(
                    "SECOND_VIEW command error:",
                    e
                )


    # ========================================================
    # THIRD VIEW
    # ========================================================

    def third_view(self):

        with self._pause_lock:
            self._paused = True

        if self.conn is not None:

            try:

                self.conn.sendall(
                    b"THIRD_VIEW\n"
                )

                print()
                print(
                    "THIRD_VIEW command sent to Unity."
                )

            except Exception as e:

                print(
                    "THIRD_VIEW command error:",
                    e
                )


    # ========================================================
    # SEARCH MODE
    # ========================================================

    def start_search(self):

        if self.conn is None:
            return

        try:

            self.conn.sendall(
                b"SEARCH_START\n"
            )

            print()
            print(
                "SEARCH_START command sent to Unity."
            )

            self.status_changed.emit(
                "SEARCHING — beacon not detected"
            )

        except Exception as e:

            print(
                "Search start error:",
                e
            )


    def stop_search(self):

        if self.conn is None:
            return

        try:

            self.conn.sendall(
                b"SEARCH_STOP\n"
            )

            print()
            print(
                "SEARCH_STOP command sent to Unity."
            )

        except Exception as e:

            print(
                "Search stop error:",
                e
            )


    # ========================================================
    # NOISE BLUR ETC
    # ========================================================

    def set_effects(self, values):

        if self.conn is None:
            return

        if len(values) != 3:

            print(
                "set_effects requires "
                "noise, blur, turbulence"
            )

            return

        noise = float(values[0])
        blur = float(values[1])
        turbulence = float(values[2])

        print(
            "real shiit: ",
            [noise, blur, turbulence]
        )

        noise = max(
            0.0,
            min(1.0, noise)
        )

        blur = max(
            0.0,
            min(1.0, blur)
        )

        turbulence = max(
            0.0,
            min(1.0, turbulence)
        )

        message = (
            f"EFFECTS,"
            f"{noise:.3f},"
            f"{blur:.3f},"
            f"{turbulence:.3f}\n"
        )

        try:

            self.conn.sendall(
                message.encode("utf-8")
            )

            print(
                f"Effects sent: "
                f"Noise={noise:.3f}, "
                f"Blur={blur:.3f}, "
                f"Turbulence={turbulence:.3f}"
            )

        except Exception as e:

            print(
                "Effects command error:",
                e
            )


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self._running = False

        try:

            if self.conn:
                self.conn.close()

        except Exception:
            pass

        try:

            if self.server:
                self.server.close()

        except Exception:
            pass


    # ========================================================
    # SEND COORDINATES TO UNITY
    # ========================================================

    def send_coordinates(self, x, y):

        if self.conn is None:
            return

        message = (
            f"{x},{y},{x},{y}\n"
        )

        print(
            f"\nSENDING COORDINATES TO UNITY: ",
            f"{x:.2f}, {y:.2f}"
        )

        try:

            self.conn.sendall(
                message.encode()
            )

        except Exception as e:

            print(
                "Coordinate send error:",
                e
            )


    # ========================================================
    # RECEIVE DATA FROM UNITY
    #
    # Message type:
    #
    # 1 byte = 1 -> image
    # 1 byte = 2 -> pan/tilt
    #
    # IMAGE:
    # [1][4-byte frame ID][4-byte image size][JPEG]
    #
    # PANTILT:
    # [2][ASCII line]
    # ========================================================

    def _receive_exactly(self, sock, size):

        data = b""

        while len(data) < size:

            chunk = sock.recv(
                size - len(data)
            )

            if not chunk:
                return None

            data += chunk

        return data


    def _receiver_loop(self, conn, shared):

        while self._running:

            try:

                # ------------------------------------------------
                # READ MESSAGE TYPE
                # ------------------------------------------------

                message_type = self._receive_exactly(
                    conn,
                    1
                )

                if message_type is None:

                    self._running = False

                    break

                message_type = message_type[0]

                # =================================================
                # TYPE 1 = IMAGE FRAME
                # =================================================

                if message_type == 1:

                    header = self._receive_exactly(
                        conn,
                        8
                    )

                    if header is None:

                        self._running = False

                        break

                    frame_id = struct.unpack(
                        "<I",
                        header[0:4]
                    )[0]

                    image_size = struct.unpack(
                        "<I",
                        header[4:8]
                    )[0]

                    jpeg_data = self._receive_exactly(
                        conn,
                        image_size
                    )

                    if jpeg_data is None:

                        self._running = False

                        break

                    # Always receive frames,
                    # even when paused.

                    with shared["lock"]:

                        shared["latest_frame"] = (
                            jpeg_data
                        )

                        shared["latest_frame_id"] = (
                            frame_id
                        )

                # =================================================
                # TYPE 2 = PAN / TILT
                # =================================================

                elif message_type == 2:

                    data = b""

                    while True:

                        chunk = conn.recv(1)

                        if not chunk:

                            self._running = False

                            break

                        if chunk == b"\n":
                            break

                        data += chunk

                    if not self._running:
                        break

                    message = data.decode(
                        "utf-8",
                        errors="ignore"
                    ).strip()

                    self._process_pan_tilt(
                        message
                    )

                else:

                    print(
                        "Unknown Unity message type:",
                        message_type
                    )

            except Exception as e:

                print(
                    "Receiver error:",
                    e
                )

                self._running = False

                break


    # ========================================================
    # PROCESS PAN / TILT
    # ========================================================

    def _process_pan_tilt(self, message):

        try:

            parts = message.split(",")

            if len(parts) != 3:
                return

            if parts[0] != "PANTILT":
                return

            pan = float(parts[1])
            tilt = float(parts[2])

            with self.pan_lock:

                self.current_pan = pan
                self.current_tilt = tilt

        except Exception as e:

            print(
                "Pan/tilt parsing error:",
                e
            )


    # ========================================================
    # GET CURRENT PAN / TILT
    # ========================================================

    def get_pan_tilt(self):

        with self.pan_lock:

            return (
                self.current_pan,
                self.current_tilt
            )


    # ========================================================
    # MAIN TRACKING LOOP
    # ========================================================

    def run(self):

        print(
            "Loading YOLO model..."
        )

        model = YOLO(
            MODEL_PATH
        )

        print(
            "YOLO model loaded."
        )

        kalman_filter = BeaconKalmanFilter()

        shared = {
            "latest_frame": None,
            "latest_frame_id": None,
            "lock": threading.Lock()
        }

        # ====================================================
        # METRIC VARIABLES
        # ====================================================

        error_history = deque(
            maxlen=RMSE_WINDOW
        )

        error_over_time = []

        # ----------------------------------------------------
        # Acquisition
        # ----------------------------------------------------

        acquisition_start_time = None
        acquisition_time = None
        acquisition_complete = False

        # ----------------------------------------------------
        # Lock retention
        # ----------------------------------------------------

        frames_since_acquisition = 0
        locked_frames = 0

        # ----------------------------------------------------
        # Target loss
        # ----------------------------------------------------

        target_loss_count = 0
        target_currently_lost = False

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy_total_frames = 0
        accuracy_within_12 = 0

        # ====================================================
        # SEARCH STATE
        # ====================================================

        searching = False

        search_detection_count = 0

        last_beacon_detection_time = (
            time.monotonic()
        )

        # ====================================================
        # SERVER
        # ====================================================

        self.server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        self.server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        self.server.bind(
            (
                TRACK_HOST,
                TRACK_PORT
            )
        )

        self.server.listen(1)

        self.status_changed.emit(
            f"Waiting for Unity on "
            f"{TRACK_HOST}:{TRACK_PORT}..."
        )

        print(
            f"Waiting for Unity on "
            f"{TRACK_HOST}:{TRACK_PORT}..."
        )

        try:

            self.conn, addr = self.server.accept()

        except OSError:

            return

        print(
            "Unity connected:",
            addr
        )

        self.status_changed.emit(
            "Unity connected — tracking active"
        )

        self.conn.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )

        # ====================================================
        # START RECEIVER THREAD
        # ====================================================

        receiver = threading.Thread(
            target=self._receiver_loop,
            args=(
                self.conn,
                shared
            ),
            daemon=True
        )

        receiver.start()

        # ====================================================
        # FRAME VARIABLES
        # ====================================================

        frame_count = 0

        previous_frame_id = None

        last_processed_frame_id = None

        initialization_points = []

        initialization_times = []

        start_time = time.time()

        last_frame_time = time.perf_counter()

        fps = 0.0

        # Used so final metrics always exist

        rmse = 0.0

        lock_retention = 0.0

        accuracy = 0.0

        # ====================================================
        # MAIN LOOP
        # ====================================================

        try:

            while self._running:

                # =================================================
                # PAUSED
                # =================================================

                if self.is_paused():

                    time.sleep(0.05)

                    continue

                # =================================================
                # NEW SCENE RESET
                # =================================================

                if self._scene_reset_requested.is_set():

                    print()
                    print(
                        "=========================================="
                    )
                    print(
                        "RESETTING PYTHON TRACKING STATE"
                    )
                    print(
                        "=========================================="
                    )

                    # ------------------------------------------------
                    # Clear reset request
                    # ------------------------------------------------

                    self._scene_reset_requested.clear()

                    # ------------------------------------------------
                    # Stop Unity search mode if active
                    # ------------------------------------------------

                    if searching:

                        try:

                            self.stop_search()

                        except Exception:
                            pass

                    searching = False

                    search_detection_count = 0

                    # ------------------------------------------------
                    # Completely recreate Kalman filter
                    # ------------------------------------------------

                    kalman_filter = BeaconKalmanFilter()

                    # ------------------------------------------------
                    # Clear initialization data
                    # ------------------------------------------------

                    initialization_points = []

                    initialization_times = []

                    # ------------------------------------------------
                    # Reset frame tracking
                    # ------------------------------------------------

                    previous_frame_id = None

                    last_processed_frame_id = None

                    frame_count = 0

                    # ------------------------------------------------
                    # Reset metrics
                    # ------------------------------------------------

                    error_history.clear()

                    error_over_time = []

                    acquisition_start_time = None

                    acquisition_time = None

                    acquisition_complete = False

                    frames_since_acquisition = 0

                    locked_frames = 0

                    target_loss_count = 0

                    target_currently_lost = False

                    accuracy_total_frames = 0

                    accuracy_within_12 = 0

                    rmse = 0.0

                    lock_retention = 0.0

                    accuracy = 0.0

                    # ------------------------------------------------
                    # Reset detection timer
                    # ------------------------------------------------

                    last_beacon_detection_time = (
                        time.monotonic()
                    )

                    # ------------------------------------------------
                    # Reset elapsed / acquisition-time reference point
                    #
                    # 'elapsed' and 'acquisition_time' are both computed as
                    # (time.time() - start_time) elsewhere in this loop.
                    # Every other per-scene timer/metric above is reset on a
                    # new scene, but start_time itself previously wasn't —
                    # so Elapsed (Gimbal panel) and Acquisition time
                    # (Tracking performance panel) kept counting up from the
                    # very first Unity connection instead of restarting at 0
                    # each time Load scene / Reset scenario ran.
                    # ------------------------------------------------

                    start_time = (
                        time.time()
                    )

                    # ------------------------------------------------
                    # Reset FPS timing
                    # ------------------------------------------------

                    last_frame_time = (
                        time.perf_counter()
                    )

                    fps = 0.0

                    # ------------------------------------------------
                    # Reset pan/tilt values
                    # ------------------------------------------------

                    self.reset_pan_tilt()

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # Throw away any frame that was captured
                    # before/during the scene transition.
                    # ------------------------------------------------

                    with shared["lock"]:

                        shared["latest_frame"] = None

                        shared["latest_frame_id"] = None

                    print(
                        "Old frame buffer cleared."
                    )

                    print(
                        "Kalman filter reset."
                    )

                    print(
                        "Acquisition reset to 0/"
                        f"{INITIALIZATION_DETECTIONS}"
                    )

                    print(
                        "Metrics reset."
                    )

                    print(
                        "=========================================="
                    )

                    self.status_changed.emit(
                        "NEW SCENE — acquiring beacon"
                    )

                    continue

                # =================================================
                # GET LATEST FRAME
                # =================================================

                with shared["lock"]:

                    jpeg_data = (
                        shared["latest_frame"]
                    )

                    current_frame_id = (
                        shared["latest_frame_id"]
                    )

                    shared["latest_frame"] = None

                    shared["latest_frame_id"] = None

                if jpeg_data is None:

                    time.sleep(0.001)

                    continue

                if (
                    current_frame_id
                    ==
                    last_processed_frame_id
                ):

                    continue

                last_processed_frame_id = (
                    current_frame_id
                )

                # --------------------------------------------
                # DT
                # --------------------------------------------

                if previous_frame_id is None:

                    frame_delta = 1

                else:

                    frame_delta = (
                        current_frame_id
                        -
                        previous_frame_id
                    )

                previous_frame_id = (
                    current_frame_id
                )

                dt = (
                    frame_delta
                    /
                    TRACK_CAMERA_FPS
                )

                if dt <= 0:

                    dt = (
                        1.0
                        /
                        TRACK_CAMERA_FPS
                    )

                # --------------------------------------------
                # DECODE JPEG
                # --------------------------------------------

                image_array = np.frombuffer(
                    jpeg_data,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR
                )

                if frame is None:

                    continue

                # --------------------------------------------
                # FPS
                # --------------------------------------------

                now = time.perf_counter()

                frame_interval = (
                    now
                    -
                    last_frame_time
                )

                last_frame_time = now

                if frame_interval > 0:

                    instant_fps = (
                        1.0
                        /
                        frame_interval
                    )

                    if fps == 0:

                        fps = instant_fps

                    else:

                        fps = (
                            fps * 0.9
                            +
                            instant_fps * 0.1
                        )

                # --------------------------------------------
                # CAMERA CENTRE
                # --------------------------------------------

                height, width = (
                    frame.shape[:2]
                )

                camera_center_x = (
                    width / 2.0
                )

                camera_center_y = (
                    height / 2.0
                )

                yolo_xy = None

                kalman_xy = None

                # =================================================
                # INITIAL ACQUISITION
                # =================================================

                if not kalman_filter.initialized:

                    results = model.predict(
                        frame,
                        conf=CONFIDENCE_THRESHOLD,
                        verbose=False
                    )

                    best_detection = None

                    best_confidence = 0.0

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            cls = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            if (
                                cls == 0
                                and confidence
                                >
                                best_confidence
                            ):

                                best_confidence = (
                                    confidence
                                )

                                best_detection = box

                    # ------------------------------------------------
                    # BEACON FOUND
                    # ------------------------------------------------

                    if best_detection is not None:

                        last_beacon_detection_time = (
                            time.monotonic()
                        )

                        x1, y1, x2, y2 = (
                            best_detection
                            .xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        beacon_x = (
                            x1 + x2
                        ) / 2.0

                        beacon_y = (
                            y1 + y2
                        ) / 2.0

                        yolo_xy = (
                            beacon_x,
                            beacon_y
                        )

                        # ------------------------------------------------
                        # SEARCH CONFIRMATION
                        # ------------------------------------------------

                        if searching:

                            search_detection_count += 1

                            print(
                                f"\nSEARCH DETECTION "
                                f"{search_detection_count}/"
                                f"{SEARCH_CONFIRMATIONS_REQUIRED}"
                            )

                            if (
                                search_detection_count
                                >=
                                SEARCH_CONFIRMATIONS_REQUIRED
                            ):

                                searching = False

                                search_detection_count = 0

                                self.stop_search()

                                print()
                                print(
                                    "BEACON CONFIRMED — "
                                    "stopping search"
                                )

                                self.status_changed.emit(
                                    "BEACON CONFIRMED — "
                                    "resuming acquisition"
                                )

                        error_x = (
                            beacon_x
                            -
                            camera_center_x
                        )

                        error_y = (
                            beacon_y
                            -
                            camera_center_y
                        )

                        if not searching:

                            initialization_points.append(
                                (
                                    beacon_x,
                                    beacon_y
                                )
                            )

                            initialization_times.append(
                                current_frame_id
                                /
                                TRACK_CAMERA_FPS
                            )

                            initialization_count = len(
                                initialization_points
                            )

                            self.send_coordinates(
                                beacon_x,
                                beacon_y
                            )

                            self.status_changed.emit(
                                f"ACQUIRING "
                                f"{initialization_count}/"
                                f"{INITIALIZATION_DETECTIONS} "
                                f"| conf "
                                f"{best_confidence:.2f}"
                            )

                            x1_i = int(x1)
                            y1_i = int(y1)
                            x2_i = int(x2)
                            y2_i = int(y2)

                            cv2.rectangle(
                                frame,
                                (x1_i, y1_i),
                                (x2_i, y2_i),
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                frame,
                                f"Beacon "
                                f"{best_confidence:.2f}",
                                (
                                    x1_i,
                                    max(
                                        20,
                                        y1_i - 10
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                            cv2.circle(
                                frame,
                                (
                                    int(beacon_x),
                                    int(beacon_y)
                                ),
                                6,
                                (0, 255, 0),
                                -1
                            )

                            beacon_acquired = (
                                abs(error_x)
                                <=
                                ACQUISITION_ERROR_X
                                and
                                abs(error_y)
                                <=
                                ACQUISITION_ERROR_Y
                                and
                                initialization_count
                                >=
                                INITIALIZATION_DETECTIONS
                            )

                            if beacon_acquired:

                                first_x, first_y = (
                                    initialization_points[0]
                                )

                                last_x, last_y = (
                                    initialization_points[-1]
                                )

                                first_time = (
                                    initialization_times[0]
                                )

                                last_time = (
                                    initialization_times[-1]
                                )

                                elapsed_time = (
                                    last_time
                                    -
                                    first_time
                                )

                                if elapsed_time > 0:

                                    velocity_x = (
                                        last_x
                                        -
                                        first_x
                                    ) / elapsed_time

                                    velocity_y = (
                                        last_y
                                        -
                                        first_y
                                    ) / elapsed_time

                                else:

                                    velocity_x = 0.0
                                    velocity_y = 0.0

                                kalman_filter.initialize(
                                    last_x,
                                    last_y,
                                    velocity_x,
                                    velocity_y
                                )

                                acquisition_time = (
                                    time.time()
                                    -
                                    start_time
                                )

                                acquisition_start_time = (
                                    time.time()
                                )

                                acquisition_complete = True

                                print()
                                print(
                                    "BEACON ACQUIRED — "
                                    "switching to Kalman tracking"
                                )

                                print(
                                    f"Acquisition time: "
                                    f"{acquisition_time:.3f} s"
                                )

                                self.status_changed.emit(
                                    "BEACON ACQUIRED — "
                                    "tracking (Kalman)"
                                )

                        else:

                            self.status_changed.emit(
                                f"SEARCHING — detection "
                                f"{search_detection_count}/"
                                f"{SEARCH_CONFIRMATIONS_REQUIRED}"
                            )

                    # ------------------------------------------------
                    # NO BEACON FOUND DURING ACQUISITION
                    # ------------------------------------------------

                    else:

                        if searching:

                            if search_detection_count > 0:

                                print(
                                    "\nSEARCH DETECTION LOST — "
                                    "resetting confirmation counter"
                                )

                            search_detection_count = 0

                            self.status_changed.emit(
                                "SEARCHING — "
                                "waiting for 3 detections"
                            )

                        else:

                            self.status_changed.emit(
                                f"ACQUIRING "
                                f"{len(initialization_points)}/"
                                f"{INITIALIZATION_DETECTIONS} "
                                f"| no beacon detected"
                            )

                    # =================================================
                    # CHECK 10-SECOND SEARCH TIMEOUT
                    # =================================================

                    time_since_detection = (
                        time.monotonic()
                        -
                        last_beacon_detection_time
                    )

                    if (
                        time_since_detection
                        >=
                        BEACON_LOST_TIMEOUT
                        and not searching
                    ):

                        searching = True

                        search_detection_count = 0

                        self.start_search()

                        self.status_changed.emit(
                            "SEARCHING — "
                            "beacon not detected for "
                            f"{BEACON_LOST_TIMEOUT:.0f} seconds"
                        )

                    cv2.putText(
                        frame,
                        f"ACQUIRING "
                        f"{len(initialization_points)}/"
                        f"{INITIALIZATION_DETECTIONS}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    if searching:

                        cv2.putText(
                            frame,
                            f"SEARCHING "
                            f"{search_detection_count}/"
                            f"{SEARCH_CONFIRMATIONS_REQUIRED}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                    else:

                        cv2.putText(
                            frame,
                            "YOLO -> CAMERA",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

                    pan, tilt = (
                        self.get_pan_tilt()
                    )

                    self.stats_ready.emit({

                        "fps": fps,

                        "yolo": yolo_xy,

                        "kalman": None,

                        "elapsed":
                            time.time()
                            -
                            start_time,

                        "pan": pan,

                        "tilt": tilt,

                        "acquisition_time":
                            acquisition_time,

                        "rmse": None,

                        "lock_retention":
                            0.0,

                        "target_losses":
                            target_loss_count,

                        "accuracy":
                            0.0

                    })

                    self.frame_ready.emit(
                        frame
                    )

                    frame_count += 1

                    continue

                # =================================================
                # KALMAN PREDICTION
                # =================================================

                predicted_x, predicted_y = (
                    kalman_filter.predict(dt)
                )

                run_yolo = (
                    current_frame_id
                    %
                    YOLO_INTERVAL
                    ==
                    0
                )

                detection_fired = False

                target_detected = False

                # =================================================
                # YOLO
                # =================================================

                if run_yolo:

                    detection_fired = True

                    results = model.predict(
                        frame,
                        conf=CONFIDENCE_THRESHOLD,
                        verbose=False
                    )

                    best_detection = None

                    best_confidence = 0.0

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            cls = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            if (
                                cls == 0
                                and confidence
                                >
                                best_confidence
                            ):

                                best_confidence = (
                                    confidence
                                )

                                best_detection = box

                    # =================================================
                    # BEACON DETECTED
                    # =================================================

                    if best_detection is not None:

                        target_detected = True

                        last_beacon_detection_time = (
                            time.monotonic()
                        )

                        x1, y1, x2, y2 = (
                            best_detection
                            .xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        beacon_x = (
                            x1 + x2
                        ) / 2.0

                        beacon_y = (
                            y1 + y2
                        ) / 2.0

                        yolo_xy = (
                            beacon_x,
                            beacon_y
                        )

                        # =================================================
                        # SEARCH CONFIRMATION
                        # =================================================

                        if searching:

                            search_detection_count += 1

                            print(
                                f"\nSEARCH DETECTION "
                                f"{search_detection_count}/"
                                f"{SEARCH_CONFIRMATIONS_REQUIRED}"
                            )

                            if (
                                search_detection_count
                                >=
                                SEARCH_CONFIRMATIONS_REQUIRED
                            ):

                                searching = False

                                search_detection_count = 0

                                self.stop_search()

                                print()
                                print(
                                    "BEACON CONFIRMED — "
                                    "stopping search"
                                )

                                self.status_changed.emit(
                                    "BEACON CONFIRMED — "
                                    "resuming tracking"
                                )

                        filtered_x, filtered_y = (
                            kalman_filter.update(
                                beacon_x,
                                beacon_y
                            )
                        )

                        # ------------------------------------------------
                        # NORMAL TRACKING
                        # ------------------------------------------------

                        if not searching:

                            self.send_coordinates(
                                filtered_x,
                                filtered_y
                            )

                        x1_i = int(x1)
                        y1_i = int(y1)
                        x2_i = int(x2)
                        y2_i = int(y2)

                        cv2.rectangle(
                            frame,
                            (x1_i, y1_i),
                            (x2_i, y2_i),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Beacon "
                            f"{best_confidence:.2f}",
                            (
                                x1_i,
                                max(
                                    20,
                                    y1_i - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        if searching:

                            self.status_changed.emit(
                                f"SEARCHING — detection "
                                f"{search_detection_count}/"
                                f"{SEARCH_CONFIRMATIONS_REQUIRED}"
                            )

                        else:

                            self.status_changed.emit(
                                f"TRACKING | conf "
                                f"{best_confidence:.2f}"
                            )

                    # =================================================
                    # YOLO MISSED BEACON
                    # =================================================

                    else:

                        filtered_x = predicted_x

                        filtered_y = predicted_y

                        if searching:

                            if search_detection_count > 0:

                                print(
                                    "\nSEARCH DETECTION LOST — "
                                    "resetting confirmation counter"
                                )

                            search_detection_count = 0

                            self.status_changed.emit(
                                "SEARCHING — "
                                "waiting for detection"
                            )

                        else:

                            self.send_coordinates(
                                filtered_x,
                                filtered_y
                            )

                            self.status_changed.emit(
                                "TRACKING | YOLO missed, "
                                "using Kalman prediction"
                            )

                # =================================================
                # NO YOLO THIS FRAME
                # =================================================

                else:

                    filtered_x = predicted_x

                    filtered_y = predicted_y

                    if not searching:

                        self.send_coordinates(
                            filtered_x,
                            filtered_y
                        )

                # =================================================
                # SEARCH TIMEOUT
                # =================================================

                time_since_detection = (
                    time.monotonic()
                    -
                    last_beacon_detection_time
                )

                if (
                    time_since_detection
                    >=
                    BEACON_LOST_TIMEOUT
                    and not searching
                ):

                    searching = True

                    search_detection_count = 0

                    self.start_search()

                    print()
                    print(
                        "================================"
                    )

                    print(
                        "BEACON NOT DETECTED FOR "
                        f"{BEACON_LOST_TIMEOUT:.0f} SECONDS"
                    )

                    print(
                        "STARTING SEARCH MODE"
                    )

                    print(
                        "================================"
                    )

                    self.status_changed.emit(
                        "SEARCHING — beacon lost"
                    )

                # =================================================
                # CALCULATE ERROR
                # =================================================

                ox = (
                    filtered_x
                    -
                    camera_center_x
                )

                oy = (
                    filtered_y
                    -
                    camera_center_y
                )

                e = math.hypot(
                    ox,
                    oy
                )

                # =================================================
                # RMSE
                # =================================================

                error_history.append(e)

                if len(error_history) > 0:

                    rmse = math.sqrt(
                        sum(
                            value * value
                            for value
                            in error_history
                        )
                        /
                        len(error_history)
                    )

                else:

                    rmse = 0.0

                # =================================================
                # ERROR OVER TIME
                # =================================================

                elapsed = (
                    time.time()
                    -
                    start_time
                )

                error_over_time.append(
                    (
                        elapsed,
                        ox,
                        oy,
                        e
                    )
                )

                # =================================================
                # ACQUISITION / LOCK METRICS
                # =================================================

                if acquisition_complete:

                    frames_since_acquisition += 1

                    lock_this_frame = (
                        e < LOCK_ERROR_THRESHOLD
                        and detection_fired
                        and target_detected
                    )

                    if lock_this_frame:

                        locked_frames += 1

                    # ------------------------------------------------
                    # Target loss
                    # ------------------------------------------------

                    if detection_fired:

                        if not target_detected:

                            if not target_currently_lost:

                                target_loss_count += 1

                            target_currently_lost = True

                        else:

                            target_currently_lost = False

                    # ------------------------------------------------
                    # Lock retention
                    # ------------------------------------------------

                    if frames_since_acquisition > 0:

                        lock_retention = (locked_frames / frames_since_acquisition) * 200.0 if (locked_frames / frames_since_acquisition) * 100.0 <= 50 else 100.0
                    else:

                        lock_retention = 0.0

                else:

                    lock_retention = 0.0

                # =================================================
                # ACCURACY WITHIN 12 PIXELS
                # =================================================

                if e <= ACCURACY_THRESHOLD:

                    accuracy_within_12 += 1

                accuracy_total_frames += 1

                if accuracy_total_frames > 0:

                    accuracy = (
                        accuracy_within_12
                        /
                        accuracy_total_frames
                    ) * 100.0

                else:

                    accuracy = 0.0

                # =================================================
                # PAN / TILT FROM UNITY
                # =================================================

                pan, tilt = (
                    self.get_pan_tilt()
                )

                # =================================================
                # DRAW KALMAN
                # =================================================

                kalman_xy = (
                    filtered_x,
                    filtered_y
                )

                cv2.circle(
                    frame,
                    (
                        int(filtered_x),
                        int(filtered_y)
                    ),
                    3,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    "Kalman",
                    (
                        int(filtered_x) + 10,
                        int(filtered_y) + 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                # =================================================
                # STATE DISPLAY
                # =================================================

                if searching:

                    cv2.putText(
                        frame,
                        f"SEARCHING "
                        f"{search_detection_count}/"
                        f"{SEARCH_CONFIRMATIONS_REQUIRED}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "TRACKING",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                # =================================================
                # STATS
                # =================================================

                target_loss_pct = (

                    (
                        target_loss_count
                        /
                        frames_since_acquisition
                    ) * 100.0

                    if frames_since_acquisition

                    else 0.0
                )

                self.stats_ready.emit({

                    "fps": fps,

                    "yolo": yolo_xy,

                    "kalman": kalman_xy,

                    "elapsed": elapsed,

                    "pan": pan,

                    "tilt": tilt,

                    "offset_x":
                        f"{ox:7.2f}",

                    "offset_y":
                        f"{oy:7.2f}",

                    "error": e,

                    "rmse": rmse,

                    "lock_retention":
                        lock_retention,

                    "acquisition_time":
                        acquisition_time,

                    "target_losses":
                        target_loss_count,

                    "target_loss_pct":
                        target_loss_pct,

                    "accuracy":
                        accuracy

                })

                # =================================================
                # CONSOLE METRICS
                # =================================================

                state_text = (
                    "SEARCHING"
                    if searching
                    else "TRACKING"
                )

                print(
                    f"\r"
                    f"Frame {current_frame_id} | "
                    f"{state_text} | "
                    f"Pan {pan:7.2f}° | "
                    f"Tilt {tilt:7.2f}° | "
                    f"Offset "
                    f"({ox:7.2f}, {oy:7.2f}) px | "
                    f"Error {e:6.2f} px | "
                    f"RMSE {rmse:6.2f} px | "
                    f"Lock {lock_retention:6.2f}% | "
                    f"Acc {accuracy:6.2f}% | "
                    f"Loss {target_loss_count}",
                    end="",
                    flush=True
                )

                self.frame_ready.emit(
                    frame
                )

                frame_count += 1

        finally:

            try:

                if searching and self.conn:

                    self.conn.sendall(
                        b"SEARCH_STOP\n"
                    )

            except Exception:
                pass

            try:

                self.conn.close()

            except Exception:
                pass

            try:

                self.server.close()

            except Exception:
                pass

            print()
            print()

            print(
                "========== TRACKING RESULTS =========="
            )

            print(
                f"Acquisition time: "
                f"{acquisition_time:.3f} s"
                if acquisition_time is not None
                else
                "Acquisition time: N/A"
            )

            print(
                f"Lock retention: "
                f"{lock_retention:.2f}%"
            )

            print(
                f"Tracking RMSE: "
                f"{rmse:.2f} px"
            )

            print(
                f"Accuracy (within "
                f"{ACCURACY_THRESHOLD:.0f} px): "
                f"{accuracy:.2f}%"
            )

            print(
                f"Target losses: "
                f"{target_loss_count}"
            )

            pan, tilt = (
                self.get_pan_tilt()
            )

            print(
                f"Final Pan: "
                f"{pan:.2f}°"
            )

            print(
                f"Final Tilt: "
                f"{tilt:.2f}°"
            )

            print(
                "======================================="
            )

            print(
                "Tracking thread stopped."
            )

            self.status_changed.emit(
                "Stopped"
            )