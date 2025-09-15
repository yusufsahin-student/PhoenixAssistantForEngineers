import webbrowser
import speech_recognition as sr
import numpy as np
import librosa
import time
import sys
import datetime
import threading
from pathlib import Path
import noisereduce as nr
import pygame
import os
from tempfile import NamedTemporaryFile
import asyncio
import edge_tts
import serial

# ---------------- Global Settings ----------------
pygame.mixer.init()  # Using "en-US-GuyNeural" voice

def remove_file_with_retry(file_path, retries=10, delay=0.1):
    for _ in range(retries):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"File {file_path} could not be removed.")

def tts_speak(text):
    async def speak_text():
        communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural", rate="+0%")
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_filename = fp.name
        await communicate.save(temp_filename)
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        remove_file_with_retry(temp_filename)
    asyncio.run(speak_text())

# ---------------- Global Variables ----------------
lock_open = False
active_user = None
note_file = Path("notes.txt")
REFERENCE_FOLDER = "references"

if not Path(REFERENCE_FOLDER).exists():
    Path(REFERENCE_FOLDER).mkdir(parents=True)

authorized_users = {}  # username (lowercase) -> reference voice file path

def load_authorized_users():
    users = {}
    for file in Path(REFERENCE_FOLDER).glob("*.wav"):
        fname = file.stem  # e.g., "reference_john"
        if fname.startswith("reference_"):
            username = fname[len("reference_"):]
            users[username.lower().strip()] = str(file)
    return users

authorized_users = load_authorized_users()

def register_reference_user():
    recognizer = sr.Recognizer()
    registered = None
    while registered is None:
        with sr.Microphone() as source:
            tts_speak("No reference voice found. Please say your name:")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        temp_file = "temp_reference.wav"
        with open(temp_file, "wb") as f:
            f.write(audio.get_wav_data())
        try:
            name = recognizer.recognize_google(audio, language="en-US").lower().strip()
            if not name:
                tts_speak("Name not detected, please try again.")
                continue
            tts_speak("Your name has been recorded as the reference.")
            new_file = Path(REFERENCE_FOLDER) / f"reference_{name}.wav"
            os.rename(temp_file, new_file)
            authorized_users[name] = str(new_file)
            registered = name
        except Exception:
            tts_speak("Reference voice not captured, please try again.")
    return registered

while not authorized_users:
    register_reference_user()
    authorized_users = load_authorized_users()

def clean_audio(file_path):
    y, sr_rate = librosa.load(file_path, sr=22050)
    y_denoised = nr.reduce_noise(y=y, sr=sr_rate)
    return y_denoised, sr_rate

def compute_mfcc(file_path):
    y, sr_rate = clean_audio(file_path)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=20)
    return np.mean(mfcc, axis=1)

def voice_similarity_check(new_file, reference_file):
    if not Path(reference_file).exists():
        tts_speak("Reference voice file not found!")
        return False
    ref_mfcc = compute_mfcc(reference_file)
    new_mfcc = compute_mfcc(new_file)
    distance = np.linalg.norm(ref_mfcc - new_mfcc)
    print(f"Voice distance: {distance}")
    return distance < 115

# ---------------- Two-Factor Verification Functions ----------------
AUTHORIZED_CODES = {
    "98765": "john",  # If the Arduino sends "98765", user "john" is accepted.
}

def authenticate_via_deneyap(port_name="COM15"):
    try:
        ser = serial.Serial(port_name, baudrate=9600, timeout=5)
        tts_speak("Deneyap board detected, performing verification...")
        time.sleep(2)
        raw_data = ser.readline()
        received_code = raw_data.decode("utf-8", errors="replace").strip()
        ser.close()
        print("Received card code:", received_code)
        if received_code in AUTHORIZED_CODES:
            global active_user
            active_user = AUTHORIZED_CODES[received_code]
            return True
        else:
            tts_speak("Invalid card code.")
            return False
    except Exception as e:
        print("Could not communicate with the Deneyap board:", e)
        tts_speak("Could not communicate with the Deneyap board.")
        return False

def voice_verification_factor():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        tts_speak("Please repeat your username for voice verification:")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            tts_speak("Voice input not detected, please try again.")
            return False
    temp_voice_file = "temp_voice.wav"
    with open(temp_voice_file, "wb") as f:
        f.write(audio.get_wav_data())
    ref_file = authorized_users.get(active_user)
    if not ref_file:
        tts_speak("Reference file not found.")
        return False
    return voice_similarity_check(temp_voice_file, ref_file)

def two_step_authentication():
    tts_speak("Please connect your Deneyap board for card verification.")
    if not authenticate_via_deneyap():
        return False
    if voice_verification_factor():
        tts_speak("Two-factor authentication successful. Unlocking.")
        global lock_open
        lock_open = True
        return True
    else:
        tts_speak("Voice verification failed.")
        return False

# ---------------- Other Functions ----------------
def add_new_user():
    global authorized_users, active_user
    recognizer = sr.Recognizer()
    tts_speak("Entering new user registration mode.")
    with sr.Microphone() as source:
        tts_speak("Please say the new user's name:")
        try:
            audio_name = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            tts_speak("No voice detected for registration.")
            return
    try:
        new_name = recognizer.recognize_google(audio_name, language="en-US").lower().strip()
        tts_speak(f"New user name: {new_name}.")
    except Exception:
        tts_speak("Could not capture the new user's name, please try again.")
        return
    if new_name in authorized_users:
        tts_speak("This user is already registered!")
        return
    with sr.Microphone() as source:
        tts_speak(f"Recording reference voice for {new_name}. Please repeat your name:")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            tts_speak("Reference voice not detected.")
            return
    temp_file = "temp_reference.wav"
    with open(temp_file, "wb") as f:
        f.write(audio.get_wav_data())
    new_file = Path(REFERENCE_FOLDER) / f"reference_{new_name}.wav"
    os.rename(temp_file, new_file)
    tts_speak("User registered successfully.")
    authorized_users[new_name] = str(new_file)

def set_alarm(time_str):
    try:
        time_str = time_str.replace('.', ':').replace(' ', ':')
        hour, minute = map(int, time_str.split(':'))
        now = datetime.datetime.now()
        alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if alarm_time < now:
            alarm_time += datetime.timedelta(days=1)
        seconds_until_alarm = (alarm_time - now).total_seconds()
        tts_speak(f"Alarm set for {hour}:{minute}.")
        def alarm_action():
            time.sleep(seconds_until_alarm)
            tts_speak("Alarm is ringing!")
        threading.Thread(target=alarm_action, daemon=True).start()
    except ValueError:
        tts_speak("Invalid time format. For example: set alarm 15:30.")

def take_note():
    tts_speak("Note taking started. Say your note. Say 'done' when finished.")
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                tts_speak("No voice detected, please try again.")
                continue
        try:
            note_text = recognizer.recognize_google(audio, language="en-US").strip()
            print("Captured note:", note_text)
            if note_text.lower() in ["done", "finished", "stop"]:
                break
            with open(note_file, "a", encoding="utf-8") as f:
                user = active_user if active_user else "User"
                f.write(f"{user}: {note_text}\n")
            tts_speak(f"Noted: {note_text}")
        except sr.UnknownValueError:
            tts_speak("Sorry, I did not understand. Please repeat.")
        except sr.RequestError:
            tts_speak("Speech service error. Please try again later.")
            break
    tts_speak("Note taking finished. Your notes have been saved.")

def voice_command():
    """
    Processes commands after the system is unlocked.
    Supported commands: "shutdown", "how are you", "date", "set alarm", "take note", "search", and "new user registration."
    """
    global lock_open, active_user, authorized_users
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            tts_speak("Waiting for your command...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
    except sr.WaitTimeoutError:
        tts_speak("No command detected, please try again.")
        return False
    try:
        command = recognizer.recognize_google(audio, language="en-US").lower().strip()
        print(f"Captured command: {command}")
    except sr.UnknownValueError:
        tts_speak("I did not catch that, please repeat.")
        return False
    except sr.RequestError:
        tts_speak("Error connecting to the speech service.")
        return False

    if lock_open and active_user is not None:
        if command == "shut down":
            tts_speak("Shutting down the system.")
            time.sleep(2)
            sys.exit()
        elif command == "how are you":
            tts_speak("I'm fine. I hope you are too!")
            return True
        elif command == "date":
            date_str = datetime.datetime.now().strftime("%d %B %Y")
            tts_speak(f"Today's date is {date_str}.")
            return True
        elif command.startswith("set alarm"):
            parts = command.split()
            if len(parts) >= 3:
                time_part = parts[-1]
                set_alarm(time_part)
                return True
            else:
                tts_speak("Please specify a valid time, for example 'set alarm 15:30'.")
                return False
        elif "search" in command:
            query = command.replace("search", "").strip()
            if query:
                tts_speak(f"Searching Google for {query}.")
                webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
                return True
            else:
                tts_speak("No search query detected.")
                return False
        elif command.startswith("take note"):
            take_note()
            return True
        elif command == "new user registration":
            add_new_user()
            return True
        else:
            tts_speak("I didn't understand that, please try again.")
            return False
    else:
        tts_speak("Please complete the authentication steps first.")
        return False

# ---------------- System Startup ----------------
tts_speak("Lock system activated. Please complete the authentication steps.")

while not lock_open:
    if two_step_authentication():
        break
    else:
        tts_speak("Authentication failed. Please try again.")

while True:
    voice_command()
