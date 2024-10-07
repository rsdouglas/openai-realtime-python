import asyncio
import websockets
import os
import json
import base64
import sounddevice as sd
import numpy as np
import threading

class AudioOut:
    def __init__(self, sample_rate, channels, output_device_id):
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_device_id = output_device_id
        self.audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        self.audio_playback_queue = asyncio.Queue()
        self.stream = None

    async def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=self._audio_callback,
            device=self.output_device_id,
            latency='low'
        )
        self.stream.start()
        await self._playback_loop()

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        bytes_to_read = frames * self.channels * 2
        with threading.Lock():
            if len(self.audio_buffer) >= bytes_to_read:
                data = self.audio_buffer[:bytes_to_read]
                del self.audio_buffer[:bytes_to_read]
            else:
                data = self.audio_buffer + bytes([0] * (bytes_to_read - len(self.audio_buffer)))
                self.audio_buffer.clear()
        outdata[:] = np.frombuffer(data, dtype='int16').reshape(-1, self.channels)

    async def _playback_loop(self):
        while True:
            chunk = await self.audio_playback_queue.get()
            if chunk is None:
                continue
            async with self.audio_buffer_lock:
                self.audio_buffer.extend(chunk)

    async def add_audio(self, chunk):
        await self.audio_playback_queue.put(chunk)

    async def clear_audio(self):
        # Clear the playback queue
        while not self.audio_playback_queue.empty():
            try:
                self.audio_playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Clear the audio buffer
        async with self.audio_buffer_lock:
            self.audio_buffer.clear()

    async def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

class AudioStreamer:
    def __init__(self, api_key, input_device_id, output_device_id):
        self.api_key = api_key
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        self.sample_rate = 24000
        self.channels = 1
        self.chunk_duration = 1
        self.audio_format = 'int16'
        self.should_record = True
        self.recorded_audio = bytearray()
        self.audio_out = AudioOut(self.sample_rate, self.channels, self.output_device_id)

    async def start(self):
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "OpenAI-Beta": "realtime=v1",
        }

        async with websockets.connect(self.url, extra_headers=headers) as ws:
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

            receive_task = asyncio.create_task(self.receive_events(ws))
            play_task = asyncio.create_task(self.audio_out.start())

            try:
                while True:
                    self.should_record = True
                    await self.send_audio(ws)
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\nExiting...")
                self.should_record = False
                receive_task.cancel()
                play_task.cancel()
                await self.audio_out.stop()
                await ws.close()

    async def send_audio(self, ws):
        print("Start speaking to the assistant (Press Ctrl+C to exit).")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time, status):
            if not self.should_record:
                return
            if status:
                print(status, flush=True)
            audio_bytes = indata.tobytes()
            self.recorded_audio.extend(audio_bytes)
            
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            message_event = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }
            
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(message_event)), loop)

        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype=self.audio_format, callback=callback, blocksize=int(self.sample_rate * self.chunk_duration), device=self.input_device_id):
            while self.should_record:
                await asyncio.sleep(1)
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

    async def receive_events(self, ws):
        while True:
            try:
                response = await ws.recv()
                event = json.loads(response)

                if event["type"] == "response.audio.delta":
                    audio_chunk = base64.b64decode(event["delta"])
                    await self.audio_out.add_audio(audio_chunk)
                elif event["type"] == "response.audio.done":
                    await self.audio_out.add_audio(None)
                    print("Response complete.")
                elif event["type"] == "input_audio_buffer.speech_started":
                    await ws.send(json.dumps({
                        "type": "response.cancel"
                    }))
                    await self.audio_out.clear_audio()
                    print("User started speaking. Clearing audio playback.")
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

def select_audio_device(input_output):
    devices = sd.query_devices()
    print(f"Available {input_output} audio devices:")
    for i, device in enumerate(devices):
        if (input_output == 'input' and device['max_input_channels'] > 0) or \
           (input_output == 'output' and device['max_output_channels'] > 0):
            print(f"{i}: {device['name']}")
    return int(input(f"Enter the number of the {input_output} device you want to use: "))

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")

    input_device_id = select_audio_device('input')
    output_device_id = select_audio_device('output')

    streamer = AudioStreamer(api_key, input_device_id, output_device_id)
    
    try:
        asyncio.run(streamer.start())
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
