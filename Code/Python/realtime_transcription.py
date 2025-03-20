# A script to demonstrate realtime (more or less...) audio transcription.
#
# Same prerequisites as whisper_large_v3.py, plus:
#
# git clone https://github.com/PortAudio/portaudio.git
# cd portaudio
# ./configure
# make -j 8
# sudo make install
# sudo cp include/* /usr/local/include # for some reason, make install does not copy all of the headers
# pip install pyaudio
# pip install wave
# pip install pydub

import pyaudio
import wave
import time
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import threading

# Select the appropriate device (MPS for Mac, CUDA for NVIDIA GPUs, or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Audio settings
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (Hz)
RECORD_SECONDS = 2  # Duration per chunk

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=0)

print("Recording started. Press Ctrl+C to stop.")

chunk_number = 0

# Define a function to run the blocking call to the audio processing model
def run_pipeline(input_data):
    result = pipe(input_data)
    print(result["text"])

# Access result after the thread finishes execution

try:
    while True:
        frames = []
        
        for _ in range(int(RECORD_SECONDS)):
            
            # capute one second of audio
            data = stream.read(RATE, exception_on_overflow=False)
            
            # Convert to NumPy array only if data is not empty
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            
            # add it to the frames list if it is not empty
            if audio_chunk.shape[0] > 0:
                frames.append(audio_chunk)
        
        if len(frames) == 0:
            print("Warning: No audio data captured!")
            continue  # Skip processing if no valid frames
        
        # Concatenate all of the frames
        audio_array = np.concatenate(frames, axis=0)
        
        # Convert int16 PCM to float64 for the transformers pipeline
        audio_array = audio_array.astype(np.float64)
        
        # Since it was recorded at 16-bit depth, dynamic range is 
        # [-32,768, 32,767].  The transformers pipeline expects values 
        # in the range of [-1,1], so we need to scale 
        audio_array = audio_array / 32768.0
        
        # tell the pipeline that it may need to resample our signal based on our native sample rate
        pipeline_input = {"sampling_rate": RATE, "raw": audio_array}
        
        # transcribe -- need to figure out how to do this in another thread so it doesn't block
        #result = pipe(pipeline_input)
        # Create and start the thread
        pipeline_thread = threading.Thread(target=run_pipeline, args=(pipeline_input, ))
        pipeline_thread.start()

        #print("Text (", chunk_number, "): ", result["text"])
        
        chunk_number += 1
except KeyboardInterrupt:
    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio stream closed.")
