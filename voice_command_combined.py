from nicegui import ui
import threading
import time
import queue
import json
import vosk
import sounddevice as sd
import pyperclip
from plyer import notification
import torch

# Загрузка модели для расстановки знаков препинания
model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_te'
)

# Глобальные переменные для управления записью
recording = False
log_queue = queue.Queue()
recognized_text_queue = queue.Queue()  # Очередь для распознанного текста
model_path = "/home/agentk/Work/VoiceCommand/vosk-model-ru-0.42"

# Глобальная переменная для модели Vosk
model = None

def load_model():
    global model
    if model is None:
        try:
            model = vosk.Model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None

q = queue.Queue()

def recognize_speech():
    load_model()
    if not model:
        log_queue.put("Model not loaded.")
        return

    rec = vosk.KaldiRecognizer(model, 16000)
    while recording:
        try:
            data = q.get(timeout=1)
        except queue.Empty:
            continue
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text = json.loads(result).get('text', '')
            if text:
                # Обрабатываем текст для расстановки знаков препинания
                text_with_punctuation = apply_te(text, lan='ru')
                log_queue.put(f"Recognized text: {text_with_punctuation}")
                recognized_text_queue.put(text_with_punctuation)  # Добавляем распознанный текст в очередь
                
                pyperclip.copy(text_with_punctuation)  # Копируем текст в буфер обмена
                notification.notify(
                    title='Text Copied',
                    message=f'{text_with_punctuation}',
                    timeout=5
                )  # Уведомление о копировании текста
        else:
            partial_result = rec.PartialResult()
            partial_text = json.loads(partial_result).get('partial', '')
            if partial_text:
                log_queue.put(f"Partial result: {partial_text}")

def start_recording():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        log_queue.put("Listening for speech...")
        while recording:
            time.sleep(1)

def callback(indata, frames, time, status):
    if status:
        log_queue.put(f"Audio callback status: {status}")
    q.put(bytes(indata))

def toggle_recording():
    global recording
    recording = not recording
    if recording:
        log_queue.put("Recording started")
        start_thread = threading.Thread(target=recognize_speech, daemon=True)
        start_thread.start()
        record_thread = threading.Thread(target=start_recording, daemon=True)
        record_thread.start()
    else:
        log_queue.put("Recording stopped")

def create_ui():
    ui.dark_mode(True)  # Включаем тёмный режим
    ui.label("Voice Command Panel").style('font-size: 24px; font-weight: bold;')
    ui.button('Toggle Recording', on_click=toggle_recording).style('font-size: 18px;')

    with ui.row():
        with ui.column().style('margin-top: 20px;').style(add="width: 30rem"):
            ui.label('Logs:').style('font-size: 18px; font-weight: bold;')

            with ui.scroll_area().style('border: 2px solid green; padding: 10px; border-radius: 5px;') as log_display:
                None

            # Обновляем содержимое текстового поля
            def update_logs():
                while True:
                    try:
                        log_message = log_queue.get(timeout=1)
                        with log_display:
                            ui.label(f"{log_message}\n")
                        # Прокручиваем к нижу
                        log_display.scroll_to(percent=100.0, axis="vertical")
                    except queue.Empty:
                        continue

            update_thread = threading.Thread(target=update_logs, daemon=True)
            update_thread.start()

        with ui.column().style('margin-top: 20px; margin-left: 20px;').style(add="width: 30rem"):
            ui.label('Recognized Text:').style('font-size: 18px; font-weight: bold;')

            recognized_text_display = ui.label().style('font-size: 16px;')

            # Обновляем содержимое распознанного текста
            def update_recognized_text():
                while True:
                    try:
                        recognized_text = recognized_text_queue.get(timeout=1)
                        recognized_text_display.set_text(f"{recognized_text}\n")
                    except queue.Empty:
                        continue

            recognized_text_thread = threading.Thread(target=update_recognized_text, daemon=True)
            recognized_text_thread.start()

    ui.run()

# Запуск интерфейса
if __name__ in {"__main__", "__mp_main__"}:
    try:
        create_ui()
        notification.notify(
            title='Voice Command Service',
            message='Сервис запущен',
            timeout=5
        )  # Уведомление о запуске сервиса
    except Exception as e:
        notification.notify(
            title='Voice Command Service',
            message=f'Ошибка: {str(e)}',
            timeout=5
        )  # Уведомление об ошибке
        raise
    finally:
        notification.notify(
            title='Voice Command Service',
            message='Сервис остановлен',
            timeout=5
        )  # Уведомление о завершении работы сервиса
