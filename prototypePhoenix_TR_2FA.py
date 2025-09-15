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

# ---------------- Global Ayarlar ----------------
pygame.mixer.init()  # "tr-TR-AhmetNeural" sesi kullanılacak

def remove_file_with_retry(file_path, retries=10, delay=0.1):
    for _ in range(retries):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"Dosya {file_path} silinemedi.")

def tts_speak(text):
    async def speak_text():
        communicate = edge_tts.Communicate(text, voice="tr-TR-AhmetNeural", rate="+0%")
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

# ---------------- Sistem Genel Değişkenleri ----------------
lock_open = False
active_user = None
note_file = Path("notlar.txt")
REFERENCE_KLASORU = "referanslar"

if not Path(REFERENCE_KLASORU).exists():
    Path(REFERENCE_KLASORU).mkdir(parents=True)

authorized_users = {}  # kullanıcı adı (küçük harf) -> referans ses yolu

def load_authorized_users():
    users = {}
    for file in Path(REFERENCE_KLASORU).glob("*.wav"):
        fname = file.stem  # örn: "referans_yusuf"
        if fname.startswith("referans_"):
            username = fname[len("referans_"):]
            users[username.lower().strip()] = str(file)
    return users

authorized_users = load_authorized_users()

def register_reference_user():
    recognizer = sr.Recognizer()
    registered = None
    while registered is None:
        with sr.Microphone() as source:
            tts_speak("Referans ses bulunamadı. Lütfen isminizi söyleyin:")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        temp_file = "temp_reference.wav"
        with open(temp_file, "wb") as f:
            f.write(audio.get_wav_data())
        try:
            name = recognizer.recognize_google(audio, language="tr-TR").lower().strip()
            if not name:
                tts_speak("İsim algılanamadı, lütfen tekrar deneyin.")
                continue
            tts_speak("Adınız referans olarak kaydedildi.")
            new_file = Path(REFERENCE_KLASORU) / f"referans_{name}.wav"
            os.rename(temp_file, new_file)
            authorized_users[name] = str(new_file)
            registered = name
        except Exception:
            tts_speak("Referans sesi kaydedilemedi, lütfen tekrar deneyin.")
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
        tts_speak("Referans ses dosyası bulunamadı!")
        return False
    ref_mfcc = compute_mfcc(reference_file)
    new_mfcc = compute_mfcc(new_file)
    distance = np.linalg.norm(ref_mfcc - new_mfcc)
    print(f"Ses uzaklığı: {distance}")
    return distance < 110

# ---------------- İki Aşamalı Doğrulama Fonksiyonları ----------------
AUTHORIZED_CODES = {
    "98765": "yusuf",  # Kart 98765 gönderdiğinde kullanıcı 'yusuf'
}

def authenticate_via_deneyap(port_name="COM15"):
    try:
        ser = serial.Serial(port_name, baudrate=9600, timeout=5)
        tts_speak("Deneyap kartı algılandı, doğrulama yapılıyor...")
        time.sleep(2)
        raw_data = ser.readline()
        received_code = raw_data.decode("utf-8", errors="replace").strip()
        ser.close()
        print("Alınan kart kodu:", received_code)
        if received_code in AUTHORIZED_CODES:
            global active_user
            active_user = AUTHORIZED_CODES[received_code]
            return True
        else:
            tts_speak("Geçersiz kart kodu.")
            return False
    except Exception as e:
        print("Deneyap kartı ile iletişim kurulamadı:", e)
        tts_speak("Deneyap kartı ile iletişim kurulamadı.")
        return False

def voice_verification_factor():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        tts_speak("Lütfen kullanıcı adınızı sesle tekrar edin:")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            tts_speak("Ses alınamadı, tekrar deneyin.")
            return False
    temp_voice_file = "temp_voice.wav"
    with open(temp_voice_file, "wb") as f:
        f.write(audio.get_wav_data())
    ref_file = authorized_users.get(active_user)
    if not ref_file:
        tts_speak("Referans dosyası bulunamadı.")
        return False
    return voice_similarity_check(temp_voice_file, ref_file)

def two_step_authentication():
    tts_speak("Lütfen Deneyap kartınızı takın ve doğrulama için bekleyin.")
    if not authenticate_via_deneyap():
        return False
    if voice_verification_factor():
        tts_speak("İki aşamalı doğrulama başarılı. Kilit açılıyor.")
        global lock_open
        lock_open = True
        return True
    else:
        tts_speak("Ses doğrulaması başarısız. Yeniden deneyin.")
        return False

# ---------------- Diğer Fonksiyonlar ----------------
def add_new_user():
    global authorized_users, active_user
    recognizer = sr.Recognizer()
    tts_speak("Yeni kullanıcı kaydı moduna giriliyor.")
    with sr.Microphone() as source:
        tts_speak("Lütfen yeni kullanıcının adını söyleyin:")
        try:
            audio_name = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            tts_speak("Kayıt için ses alınamadı.")
            return
    try:
        new_name = recognizer.recognize_google(audio_name, language="tr-TR").lower().strip()
        tts_speak(f"Yeni kullanıcı adı: {new_name}.")
    except Exception:
        tts_speak("Yeni kullanıcının adı yakalanamadı, lütfen tekrar deneyin.")
        return
    if new_name in authorized_users:
        tts_speak("Bu kullanıcı zaten kayıtlı!")
        return
    with sr.Microphone() as source:
        tts_speak(f"{new_name} için referans sesi kaydediliyor. Lütfen adınızı tekrar edin:")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            tts_speak("Referans sesi alınamadı.")
            return
    temp_file = "temp_reference.wav"
    with open(temp_file, "wb") as f:
        f.write(audio.get_wav_data())
    new_file = Path(REFERENCE_KLASORU) / f"referans_{new_name}.wav"
    os.rename(temp_file, new_file)
    tts_speak("Kullanıcı başarıyla kaydedildi.")
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
        tts_speak(f"Alarm {hour}:{minute} olarak ayarlandı.")
        def alarm_action():
            time.sleep(seconds_until_alarm)
            tts_speak("Alarm çalıyor!")
        threading.Thread(target=alarm_action, daemon=True).start()
    except ValueError:
        tts_speak("Geçersiz zaman formatı. Örneğin: alarm kur 15:30.")

def take_note():
    tts_speak("Not almaya başlıyoruz. Lütfen eklemek istediğiniz notları söyleyin; bitirmek için 'bitti' deyin.")
    recognizer = sr.Recognizer()
    full_note = ""
    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                continue
        try:
            note_part = recognizer.recognize_google(audio, language="tr-TR").lower().strip()
            print("Alınan not bölümü:", note_part)
            if note_part == "bitti":
                break
            else:
                full_note += note_part + " "
        except sr.UnknownValueError:
            tts_speak("Anlayamadım, lütfen tekrar edin.")
        except sr.RequestError:
            tts_speak("Not alınırken hata oluştu.")
    if full_note.strip():
        with open(note_file, "a", encoding="utf-8") as f:
            f.write(f"{active_user if active_user else 'Kullanıcı'}: {full_note.strip()}\n")
        tts_speak("Notunuz kaydedildi.")
    else:
        tts_speak("Herhangi bir not alınamadı.")

def voice_command():
    """
    Kilit açıldıktan sonra, kullanıcının sesli komutlarını işler.
    Desteklenen komutlar: "sistem kapat", "nasılsın", "tarih", "alarm kur", "not al", "ara", "yeni kullanıcı kaydı".
    """
    global lock_open, active_user, authorized_users
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            tts_speak("Komut bekleniyor...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
    except sr.WaitTimeoutError:
        tts_speak("Komut alınamadı, lütfen tekrar deneyin.")
        return False
    try:
        command = recognizer.recognize_google(audio, language="tr-TR").lower().strip()
        print(f"Tanınan komut: {command}")
    except sr.UnknownValueError:
        tts_speak("Anlayamadım, lütfen tekrar edin.")
        return False
    except sr.RequestError:
        tts_speak("Konuşma servisine ulaşılamadı.")
        return False

    if lock_open and active_user is not None:
        if command == "sistem kapat":
            tts_speak("Sistem kapatılıyor.")
            time.sleep(2)
            sys.exit()
            return True
        elif command == "tarih":
            date_str = datetime.datetime.now().strftime("%d %B %Y")
            tts_speak(f"Bugünün tarihi {date_str}.")
            return True
        elif command.startswith("alarm kur"):
            parts = command.split()
            if len(parts) >= 3:
                time_part = parts[-1]
                set_alarm(time_part)
                return True
            else:
                tts_speak("Lütfen geçerli bir zaman belirtin, örneğin 'alarm kur 15:30'.")
                return False
        elif "ara" in command:
            query = command.replace("ara", "").strip()
            if query:
                tts_speak(f"{query} için Google'da arama yapılıyor.")
                webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
                return True
            else:
                tts_speak("Aranacak ifade bulunamadı.")
                return False
        elif command.startswith("not al"):
            take_note()
            return True
        elif command == "yeni kullanıcı kaydı":
            add_new_user()
            return True
        else:
            tts_speak("Bunu anlayamadım.")
            return False
    else:
        tts_speak("Lütfen önce doğrulama adımlarını tamamlayın.")
        return False

# ---------------- Sistem Başlangıcı ----------------
tts_speak("Kilit sistemi etkinleştirildi. Lütfen doğrulama adımlarını takip edin.")

while not lock_open:
    if two_step_authentication():
        break
    else:
        tts_speak("Doğrulama başarısız. Lütfen tekrar deneyin.")

while True:
    voice_command()
