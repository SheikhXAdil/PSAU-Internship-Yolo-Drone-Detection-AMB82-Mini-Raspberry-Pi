import cv2
import numpy as np
import ncnn
import threading
import time

from flask import Flask, Response


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PARAM = "/home/pi/drone_detection/model.ncnn.param"
MODEL_BIN = "/home/pi/drone_detection/model.ncnn.bin"

INPUT_BLOB = "in0"
OUTPUT_BLOB = "out0"

INPUT_SIZE = 320

CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.55

# Run YOLO inference every Nth frame
FRAME_SKIP = 5

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480 

STREAM_PORT = 8080

CLASS_NAMES = [
    "drone"
]

CAMERA_DEVICE = "/dev/video1"



# ============================================================
# GLOBAL STREAM FRAME
# ============================================================

latest_frame = None
frame_lock = threading.Lock()


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(image, new_size=320):

    h, w = image.shape[:2]

    scale = min(
        new_size / w,
        new_size / h
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_w = new_size - new_w
    pad_h = new_size - new_h

    left = pad_w // 2
    right = pad_w - left

    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return padded, scale, left, top


# ============================================================
# NCNN MODEL
# ============================================================

net = ncnn.Net()

# CPU inference
net.opt.use_vulkan_compute = False

# Number of CPU threads
net.opt.num_threads = 4

print("Loading NCNN model...")

ret = net.load_param(MODEL_PARAM)

if ret != 0:
    raise RuntimeError(
        f"Could not load param file: {MODEL_PARAM}"
    )

ret = net.load_model(MODEL_BIN)

if ret != 0:
    raise RuntimeError(
        f"Could not load bin file: {MODEL_BIN}"
    )

print("NCNN model loaded.")


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_inference(frame):

    original_h, original_w = frame.shape[:2]

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    image, scale, pad_x, pad_y = letterbox(
        frame,
        INPUT_SIZE
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # # uint8 -> float32
    # image = image.astype(np.float32)

    # # Normalize 0-255 -> 0-1
    # image /= 255.0

    # --------------------------------------------------------
    # NCNN INPUT
    # --------------------------------------------------------

    # NCNN expects HWC RGB image through from_pixels
    mat = ncnn.Mat.from_pixels(
        image,
        ncnn.Mat.PixelType.PIXEL_RGB,
        INPUT_SIZE,
        INPUT_SIZE
    )

    mat.substract_mean_normalize(
        [0, 0, 0],
        [1 / 255.0, 1 / 255.0, 1 / 255.0]
    )

    extractor = net.create_extractor()

    extractor.set_light_mode(True)

    ret = extractor.input(
        INPUT_BLOB,
        mat
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN input failed: {ret}"
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    ret, output = extractor.extract(
        OUTPUT_BLOB
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN inference failed: {ret}"
        )

    # Convert NCNN tensor to numpy
    detections = np.array(output)

    return detections, scale, pad_x, pad_y

# ============================================================
# POSTPROCESS YOLO OUTPUT
# ============================================================

# ============================================================
# POSTPROCESS YOLO11 OUTPUT
# ============================================================

def postprocess(
    frame,
    detections,
    scale,
    pad_x,
    pad_y
):

    boxes = []
    scores = []
    class_ids = []

    original_h, original_w = frame.shape[:2]

    detections = np.asarray(
        detections,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # YOLO11 NCNN output:
    #
    # Shape:
    #     (5, 2100)
    #
    # Rows:
    #     0 -> cx
    #     1 -> cy
    #     2 -> width
    #     3 -> height
    #     4 -> class confidence
    #
    # Columns:
    #     2100 candidate detections
    # --------------------------------------------------------

    if detections.ndim != 2:
        print(
            "Unexpected output shape:",
            detections.shape
        )
        return []

    if detections.shape[0] != 5:
        print(
            "Unexpected YOLO11 output shape:",
            detections.shape
        )
        return []

    # --------------------------------------------------------
    # Process 2100 detections
    # --------------------------------------------------------

    for i in range(
        detections.shape[1]
    ):

        # ----------------------------------------------------
        # Extract detection
        # ----------------------------------------------------

        cx = float(
            detections[0, i]
        )

        cy = float(
            detections[1, i]
        )

        w = float(
            detections[2, i]
        )

        h = float(
            detections[3, i]
        )

        confidence = float(
            detections[4, i]
        )

        # ----------------------------------------------------
        # Confidence threshold
        # ----------------------------------------------------

        if confidence < CONF_THRESHOLD:
            continue

        # Single class: drone
        class_id = 0

        # ----------------------------------------------------
        # Center -> corner coordinates
        # ----------------------------------------------------

        x1 = cx - w / 2
        y1 = cy - h / 2

        x2 = cx + w / 2
        y2 = cy + h / 2

        # ----------------------------------------------------
        # Undo letterbox
        # ----------------------------------------------------

        x1 = (
            x1 - pad_x
        ) / scale

        y1 = (
            y1 - pad_y
        ) / scale

        x2 = (
            x2 - pad_x
        ) / scale

        y2 = (
            y2 - pad_y
        ) / scale

        # ----------------------------------------------------
        # Clamp to original camera image
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                original_w - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                original_h - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                original_w - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                original_h - 1,
                y2
            )
        )

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 1 or box_h <= 1:
            continue

        # ----------------------------------------------------
        # Store detection
        # ----------------------------------------------------

        boxes.append([
            int(x1),
            int(y1),
            int(box_w),
            int(box_h)
        ])

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if len(boxes) == 0:
        return []

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    results = []

    if len(indices) > 0:

        indices = np.array(
            indices
        ).flatten()

        for i in indices:

            x, y, w, h = boxes[i]

            results.append({

                "box": (
                    x,
                    y,
                    w,
                    h
                ),

                "confidence": scores[i],

                "class_id": class_ids[i]

            })

    return results

# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    frame,
    detections
):

    for detection in detections:

        x, y, w, h = detection["box"]

        confidence = detection["confidence"]

        class_id = detection["class_id"]

        if class_id < len(CLASS_NAMES):

            class_name = CLASS_NAMES[class_id]

        else:

            class_name = str(class_id)

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        # Bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Text background
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        cv2.rectangle(
            frame,
            (x, y - th - baseline - 5),
            (x + tw + 5, y),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x + 2, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return frame


# ============================================================
# USB CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

# USB camera supports MJPEG at 640x480 @ 30 FPS
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open USB camera: {CAMERA_DEVICE}"
    )

print(
    f"USB camera started: "
    f"{CAMERA_DEVICE} "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
)


# ============================================================
# DETECTION LOOP
# ============================================================

def detection_loop():

    global latest_frame

    frame_count = 0

    # Last successful detection result
    last_detections = []

    # Last inference information
    last_inference_time = 0.0

    # FPS measurement
    fps_start = time.time()

    processed_frames = 0
    displayed_frames = 0

    inference_fps = 0.0
    stream_fps = 0.0

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("USB camera frame capture failed")
            continue

        frame_count += 1
        displayed_frames += 1

        # ----------------------------------------------------
        # Run YOLO only every Nth frame
        # ----------------------------------------------------

        if frame_count % FRAME_SKIP == 0:

            inference_start = time.time()

            try:

                output, scale, pad_x, pad_y = run_inference(
                    frame
                )

                detections = postprocess(
                    frame,
                    output,
                    scale,
                    pad_x,
                    pad_y
                )

                # Save latest successful detections
                last_detections = detections

            except Exception as e:

                print(
                    "Inference error:",
                    e
                )

                # Keep previous detections instead of
                # immediately removing them
                detections = last_detections

            last_inference_time = (
                time.time() -
                inference_start
            )

            processed_frames += 1

        else:

            # No inference on this frame.
            # Reuse the previous detection.
            detections = last_detections

        # ----------------------------------------------------
        # FPS calculation
        # ----------------------------------------------------

        elapsed = time.time() - fps_start

        if elapsed >= 1.0:

            # Camera/display loop FPS
            stream_fps = displayed_frames / elapsed

            # Actual YOLO inference FPS
            inference_fps = processed_frames / elapsed

            displayed_frames = 0
            processed_frames = 0

            fps_start = time.time()

        # ----------------------------------------------------
        # Draw results
        # ----------------------------------------------------

        frame = draw_detections(
            frame,
            detections
        )

        # ----------------------------------------------------
        # Performance overlay
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Stream FPS: {stream_fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inference FPS: {inference_fps:.1f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inference: {last_inference_time * 1000:.1f} ms",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Objects: {len(detections)}",
            (10, 109),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Skip: {FRAME_SKIP}",
            (10, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # JPEG encode
        # ----------------------------------------------------

        success, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:
            continue

        # ----------------------------------------------------
        # Update stream frame
        # ----------------------------------------------------

        with frame_lock:

            latest_frame = jpeg.tobytes()

# ============================================================
# MJPEG STREAM
# ============================================================

app = Flask(__name__)


def generate_stream():

    global latest_frame

    while True:

        with frame_lock:

            frame = latest_frame

        if frame is None:

            time.sleep(0.01)

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.001)


@app.route("/")
def index():

    return """
    <html>

    <head>
        <title>Raspberry Pi YOLOv7</title>
    </head>

    <body>

        <h1>YOLOv7 Drone Detection</h1>

        <img
            src="/stream.mjpg"
            width="640"
        >

    </body>

    </html>
    """


@app.route("/stream.mjpg")
def stream():

    return Response(
        generate_stream(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    detection_thread = threading.Thread(
        target=detection_loop,
        daemon=True
    )

    detection_thread.start()

    print()
    print("========================================")
    print(" Raspberry Pi YOLOv7 NCNN Detector")
    print("========================================")
    print()
    print(
        f"Open VLC/network stream:"
    )
    print()
    print(
        f"http://<RASPBERRY_PI_IP>:{STREAM_PORT}/stream.mjpg"
    )
    print()
    print(
        "Browser:"
    )
    print(
        f"http://<RASPBERRY_PI_IP>:{STREAM_PORT}/"
    )
    print()
    print("Press Ctrl+C to stop.")
    print()

    app.run(
        host="0.0.0.0",
        port=STREAM_PORT,
        threaded=True
    )
