import cv2
import torch
import pyttsx3
import threading
import queue
import time

# Initialize text-to-speech engine in the main thread
engine = pyttsx3.init()
engine.setProperty('rate', 175)  # Adjust speech rate
engine.setProperty('volume', 1.0)  # Set volume to maximum

# Load YOLOv5 model (using CPU if CUDA is not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.conf = 0.5  # Set confidence threshold
model.iou = 0.4  # Set IoU threshold

# Create queues for frames and audio output
frame_queue = queue.Queue(maxsize=2)
audio_queue = queue.Queue()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS for Raspberry Pi performance

# Frame skipping parameters
frame_skip = 2  # Skip every alternate frame
frame_count = 0

# Track detected objects
active_objects = set()

# Known parameters for distance estimation
FOCAL_LENGTH = 600  # Pre-calibrated focal length (in pixels)
KNOWN_WIDTH = {
    "person": 50,  # Known widths of objects (in cm)
    "chair": 40,
    "bottle": 8,
    "cup": 7,
    "car": 150
}

# Stop event for graceful termination
stop_event = threading.Event()

def calculate_distance(real_width, pixel_width):
    """Calculate distance using the monocular method."""
    if pixel_width > 0:
        return round((real_width * FOCAL_LENGTH) / pixel_width, 2)
    return -1  # Return -1 if calculation is invalid

def capture_frames():
    """Capture frames from the webcam and put them into the queue."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    """Process frames for object detection and update detected objects."""
    global active_objects, frame_count

    while not stop_event.is_set():
        if frame_queue.empty():
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames for performance
            continue

        frame = frame_queue.get()

        # Convert frame to RGB for YOLO processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
        results = model(rgb_frame, size=320)
        detected_objects = results.xyxy[0]

        # Process detections
        current_objects = set()
        for det in detected_objects:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = f"{results.names[int(cls)]}"
            pixel_width = int(x2 - x1)

            # Calculate distance if the object has a known real-world width
            if label in KNOWN_WIDTH:
                distance = calculate_distance(KNOWN_WIDTH[label], pixel_width)
                distance_message = f"{label} at {distance} centimeters"
            else:
                distance_message = f"{label} detected"  # For unknown objects

            current_objects.add((label, distance_message))

            # Draw bounding box and display distance
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{distance_message} ({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Speak only newly detected objects
            if (label, distance_message) not in active_objects and distance_message not in audio_queue.queue:
                print(f"Queuing Speech: {distance_message}")
                audio_queue.put(distance_message)

        # Update active objects
        active_objects = current_objects

        # Display frame
        cv2.imshow('YOLOv5 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

def speak_objects():
    """Continuously process audio queue to speak detected objects."""
    while not stop_event.is_set():
        if not audio_queue.empty():
            message = audio_queue.get()
            print(f"Speaking: {message}")
            engine.say(f"Detected {message}")
            engine.runAndWait()  # Run in main thread to avoid 'QObject::startTimer' error

def main():
    # Start threads for capturing frames and processing detection
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=process_frames, daemon=True).start()

    try:
        # Run speech synthesis in the main thread
        speak_objects()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping program...")
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
