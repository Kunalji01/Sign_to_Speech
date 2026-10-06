import sys
import cv2
import mediapipe as mp
import csv
import os
import time

# ==============================
# MEDIAPIPE SETUP
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ==============================
# GET SIGN NAME
# ==============================
if len(sys.argv) < 2:
    print("Usage: python collect_data.py SIGN_NAME")
    exit()

sign_name = sys.argv[1].lower()

# ==============================
# DATASET SETUP
# ==============================
os.makedirs("dataset", exist_ok=True)

csv_file = f"dataset/{sign_name}.csv"

print(f"Collecting data for: {sign_name.upper()}")

# Count existing samples
sample_count = 0

if os.path.exists(csv_file):
    with open(csv_file, "r") as file:
        sample_count = sum(1 for _ in file)

print(f"Existing samples: {sample_count}")

# ==============================
# CAMERA SETUP
# ==============================
camera = cv2.VideoCapture(0)

# Debounce control
last_save_time = 0
SAVE_DELAY = 0.5


# ==============================
# MAIN LOOP
# ==============================
with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera read failed")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        # ==============================
        # DISPLAY INFORMATION
        # ==============================
        cv2.putText(
            frame,
            f"Sign: {sign_name.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press S = Save | Q = Quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Show hand detection status
        if result.hand_landmarks:
            status = "Hand Detected"
        else:
            status = "No Hand Detected"

        cv2.putText(
            frame,
            status,
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Show camera FIRST
        cv2.imshow(
            f"Collect {sign_name.upper()} Data",
            frame
        )

        # Read keyboard
        key = cv2.waitKey(1) & 0xFF

        # ==============================
        # SAVE DATA
        # ==============================
        if (
            key == ord("s")
            and time.time() - last_save_time > SAVE_DELAY
        ):

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # Wrist landmark = reference point
                wrist = hand[0]

                row = []

                # Normalize all landmarks relative to wrist
                for landmark in hand:

                    row.append(
                        landmark.x - wrist.x
                    )

                    row.append(
                        landmark.y - wrist.y
                    )

                    row.append(
                        landmark.z - wrist.z
                    )

                # Save exactly ONE row
                with open(
                    csv_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)
                    writer.writerow(row)

                sample_count += 1
                last_save_time = time.time()

                print(
                    f"Saved sample {sample_count}"
                )

            else:
                print(
                    "No hand detected! Sample not saved."
                )

        # ==============================
        # QUIT
        # ==============================
        if key == ord("q"):
            break


# ==============================
# CLEANUP
# ==============================
camera.release()
cv2.destroyAllWindows()