import cv2
import mediapipe as mp

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

camera = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            print("Camera nahi mil raha")
            break

        # OpenCV BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Hand detection
        result = landmarker.detect(mp_image)

        # Draw detected landmarks
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                print("Number of landmarks:", len(hand))

            for i, landmark in enumerate(hand):
                print(
                    i,
                    "x:", round(landmark.x, 3),
                    "y:", round(landmark.y, 3),
                    "z:", round(landmark.z, 3)
    )

                print("-------------------")

                height, width, _ = frame.shape

                # Points
                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Connections
                connections = [
                    (0,1), (1,2), (2,3), (3,4),
                    (0,5), (5,6), (6,7), (7,8),
                    (5,9), (9,10), (10,11), (11,12),
                    (9,13), (13,14), (14,15), (15,16),
                    (13,17), (17,18), (18,19), (19,20),
                    (0,17)
                ]

                for start, end in connections:
                    x1 = int(hand[start].x * width)
                    y1 = int(hand[start].y * height)

                    x2 = int(hand[end].x * width)
                    y2 = int(hand[end].y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("SignBridge - Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()