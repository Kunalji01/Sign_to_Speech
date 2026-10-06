import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

print("Speaking now...")

engine.say("Hello. This is a speech test.")
engine.runAndWait()

print("Finished")