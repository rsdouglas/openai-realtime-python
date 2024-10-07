import asyncio
import websockets
import os
import json
import base64
import sounddevice as sd
import numpy as np
import threading

# Set the API endpoint and model
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Get the API key from environment variable or prompt
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Please enter your OpenAI API key: ")

# Prompt user to select input device
devices = sd.query_devices()
print("Available input audio devices:")
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"{i}: {device['name']}")

input_device_id = int(input("Enter the number of the input device you want to use: "))

print("Available output audio devices:")
for i, device in enumerate(devices):
    if device['max_output_channels'] > 0:
        print(f"{i}: {device['name']}")

output_device_id = int(input("Enter the number of the output device you want to use: "))

# Audio parameters
SAMPLE_RATE = 24000  # 24 kHz
CHANNELS = 1
CHUNK_DURATION = 1  # Duration of each audio chunk in seconds
AUDIO_FORMAT = 'int16'  # 16-bit PCM audio
sd.default.channels = CHANNELS

# Flags for controlling recording and playback
should_record = True
audio_playback_queue = asyncio.Queue()
recorded_audio = bytearray()
audio_buffer = bytearray()
audio_buffer_lock = threading.Lock()

async def send_audio(ws):
    global should_record
    print("Start speaking to the assistant (Press Ctrl+C to exit).")
    loop = asyncio.get_event_loop()

    def callback(indata, frames, time, status):
        if not should_record:
            return
        if status:
            print(status, flush=True)
        audio_bytes = indata.tobytes()
        recorded_audio.extend(audio_bytes)
        
        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
        message_event = {
            "type": "input_audio_buffer.append",
            "audio": encoded_audio
        }
        
        asyncio.run_coroutine_threadsafe(ws.send(json.dumps(message_event)), loop)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=AUDIO_FORMAT, callback=callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION), device=input_device_id):
        while should_record:
            await asyncio.sleep(1)
            await ws.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))

async def receive_events(ws):
    global audio_buffer
    while True:
        try:
            response = await ws.recv()
            event = json.loads(response)

            if event["type"] == "response.audio.delta":
                audio_chunk = base64.b64decode(event["delta"])
                await audio_playback_queue.put(audio_chunk)
            elif event["type"] == "response.audio.done":
                await audio_playback_queue.put(None)
                print("Response complete.")
            elif event["type"] == "input_audio_buffer.speech_started":
                await ws.send(json.dumps({
                    "type": "response.cancel"
                }))
                with audio_buffer_lock:
                    audio_buffer.clear()
                print("User started speaking.")
            elif event["type"] == "input_audio_buffer.speech_stopped":
                print("User stopped speaking.")
            elif event["type"] == "error":
                error = event.get("error", {})
                message = error.get("message", "")
                if message != "Error committing input audio buffer: the buffer is empty.":
                    print(f"Error: {message}")
            else:
                pass
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            break

async def play_audio():
    global audio_buffer
    global audio_buffer_lock
    
    def callback(outdata, frames, time, status):
        global audio_buffer
        if status:
            print(status)
        bytes_to_read = frames * CHANNELS * 2
        with audio_buffer_lock:
            if len(audio_buffer) >= bytes_to_read:
                data = audio_buffer[:bytes_to_read]
                del audio_buffer[:bytes_to_read]
            else:
                data = audio_buffer + bytes([0] * (bytes_to_read - len(audio_buffer)))
                audio_buffer.clear()
        outdata[:] = np.frombuffer(data, dtype=AUDIO_FORMAT).reshape(-1, CHANNELS)

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=AUDIO_FORMAT,
        callback=callback,
        device=output_device_id,
        latency='low'
    ):
        while True:
            chunk = await audio_playback_queue.get()
            if chunk is None:
                continue
            else:
                with audio_buffer_lock:
                    audio_buffer.extend(chunk)

async def main():
    headers = {
        "Authorization": "Bearer " + api_key,
        "OpenAI-Beta": "realtime=v1",
    }

    global should_record

    async with websockets.connect(url, extra_headers=headers) as ws:
        print("Connected to the OpenAI Realtime API.")

        event = await ws.recv()
        event_data = json.loads(event)
        if event_data["type"] == "session.created":
            print("Session initialized.")

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "server_vad"
                },
            }
        }))

        receive_task = asyncio.create_task(receive_events(ws))
        play_task = asyncio.create_task(play_audio())

        try:
            while True:
                should_record = True
                await send_audio(ws)
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
            should_record = False
            receive_task.cancel()
            play_task.cancel()
            await ws.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {str(e)}")
