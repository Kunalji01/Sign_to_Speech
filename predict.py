import cv2
import mediapipe as mp
import joblib
import numpy as np
from gtts import gTTS
import os
from playsound3 import playsound
import subprocess


# ================= LOAD MODEL =================
model = joblib.load("models/sign_model.pkl")


# ================= TRANSLATIONS =================
hindi_translation = {
    "hello": "नमस्ते",
    "help": "मदद",
    "no": "नहीं",
    "please": "कृपया",
    "sorry": "माफ़ कीजिए",
    "stop": "रुकिए",
    "thankyou": "धन्यवाद",
    "yes": "हाँ",
    "good": "अच्छा",
    "bad": "बुरा",
    "welcome": "स्वागत है",
    "morning": "सुबह",
    "night": "रात",
    "eat": "खाना",
    "drink": "पीना",
    "water": "पानी"
}


# ================= LANGUAGE MODE =================
language_mode = "english"


# ================= TEXT TO SPEECH =================
def speak(text, language):

    # -------- HINDI --------
    if language == "hindi":

        output_text = hindi_translation.get(
            text.lower(),
            text
        )

        print("LANGUAGE: HINDI")
        print("SPEAKING:", output_text)

        try:
            tts = gTTS(
                text=output_text,
                lang="hi",
                slow=False
            )

            audio_file = "temp_hindi.mp3"

            tts.save(audio_file)

            playsound(audio_file)

            if os.path.exists(audio_file):
                os.remove(audio_file)

        except Exception as e:
            print("Hindi Speech Error:", e)


    # -------- ENGLISH --------
    else:

        output_text = text

        print("LANGUAGE: ENGLISH")
        print("SPEAKING:", output_text)

        command = f'''
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 0
$speaker.Volume = 100
$speaker.Speak("{output_text}")
'''

        try:
            subprocess.run([
                "powershell",
                "-Command",
                command
            ])

        except Exception as e:
            print("English Speech Error:", e)


# ================= MEDIAPIPE =================
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


# ================= CAMERA =================
camera = cv2.VideoCapture(0)


# ================= SPEECH VARIABLES =================
current_sign = None
last_spoken = None
stable_count = 0

STABLE_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.80


# ================= MAIN PROGRAM =================
with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera nahi mil raha")
            break


        # Convert frame to RGB
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

        prediction_text = "No Hand"
        hindi_text = ""


        # =====================================
        # HAND DETECTED
        # =====================================
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Normalize landmarks relative to wrist
            wrist = hand[0]

            features = []

            for landmark in hand:

                x = landmark.x - wrist.x
                y = landmark.y - wrist.y
                z = landmark.z - wrist.z

                features.extend([x, y, z])


            # Convert features
            features = np.array(features).reshape(1, -1)


            # ML Prediction
            prediction = model.predict(features)[0]

            probability = model.predict_proba(
                features
            ).max()


            # Hindi translation
            hindi_text = hindi_translation.get(
                prediction.lower(),
                prediction
            )


            # Prediction display
            prediction_text = (
                f"{prediction.upper()} "
                f"({probability * 100:.1f}%)"
            )


            # =====================================
            # STABLE DETECTION
            # =====================================
            if probability >= CONFIDENCE_THRESHOLD:

                if prediction == current_sign:
                    stable_count += 1

                else:
                    current_sign = prediction
                    stable_count = 1


                # Speak stable new sign
                if (
                    stable_count >= STABLE_FRAMES
                    and prediction != last_spoken
                ):

                    print("NEW SIGN:", prediction)

                    last_spoken = prediction

                    speak(
                        prediction,
                        language_mode
                    )

                    stable_count = 0

            else:
                stable_count = 0


        # =====================================
        # NO HAND
        # =====================================
        else:

            current_sign = None
            stable_count = 0
            last_spoken = None


        # =====================================
        # DISPLAY
        # =====================================

        cv2.putText(
            frame,
            prediction_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Language: {language_mode.upper()}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "E: English | H: Hindi | Q: Quit",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Stable: {stable_count}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "SignBridge - Multilingual Sign to Speech",
            frame
        )


        # =====================================
        # KEYBOARD CONTROLS
        # =====================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("e"):

            language_mode = "english"

            # Allow current sign to speak again
            last_spoken = None
            stable_count = 0

            print("\nLanguage switched to ENGLISH 🇬🇧")


        elif key == ord("h"):

            language_mode = "hindi"

            # Allow current sign to speak again
            last_spoken = None
            stable_count = 0

            print("\nLanguage switched to HINDI 🇮🇳")


        elif key == ord("q"):
            break


# ================= CLEANUP =================
camera.release()
cv2.destroyAllWindows()