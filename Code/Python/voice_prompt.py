# A simple python program that records audio from a microphone on the machine where 
# the program is run, uses OpenAI's whisper ASR model to transcribe the audio, 
# then passes the transcribed text as a prompt to an LLM and prints the response.

import pyaudio
import wave
import time
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import threading
from openai import OpenAI
import warnings

#---------------------------------------------------------------------------
# Settings:

# which LLM should we use?
#llm_model_id = "Llama-3.3-70B-Instruct:latest"
#llm_model_id = "gemma3:27b"
llm_model_id = "phi4:14b-fp16"

# where is the LLM hosted (via ollama)?
llm_host_address = "localhost"

# what model should we use for automatic speech recognition?
asr_model_id = "openai/whisper-large-v3"

# what is the pyaudio device index of the microphone that you would like to use?
input_device_index = 0
#---------------------------------------------------------------------------

# there is a parameter name change that we cannot control causing warnings every time the whisper model is called
warnings.filterwarnings('ignore')

# Select the appropriate device (MPS for Mac, CUDA for NVIDIA GPUs, or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    asr_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(asr_model_id)
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
                input_device_index=input_device_index)

# record audio and send to the LLM the contents of the conversation argument concatenated with the 
# transcription of the recording
def record_and_prompt(conversation, stream):
    
    print("Recording started. Press Ctrl+C to stop.")
    frames = []
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            if audio_chunk.shape[0] > 0:
                frames.append(audio_chunk)
    
    except KeyboardInterrupt:
        print("Recording stopped.")
    
    # Concatenate all of the audio frames
    audio_array = np.concatenate(frames, axis=0)
    
    # Convert int16 PCM to float64 for the transformers pipeline
    audio_array = audio_array.astype(np.float64)
    
    # Since it was recorded at 16-bit depth, dynamic range is
    # [-32,768, 32,767].  The transformers pipeline expects values
    # in the range of [-1,1], so we need to scale
    audio_array = audio_array / 32768.0
    
    # format input for the pipeline
    pipeline_input = {"sampling_rate": RATE, "raw": audio_array}
    
    # run the model to transcribe the recording
    result = pipe(pipeline_input)
    transcribed_text = result["text"]
    
    # print the transcribed text to the console
    print("Transcribed Text: ", transcribed_text)
    
    # Set up the OpenAI API client
    client = OpenAI(base_url="http://" + llm_host_address + ":11434/v1", api_key="not-used")
    
    # send the previous conversation plus the transcribed text to the LLM
    stream = client.chat.completions.create(
        model=llm_model_id,
        messages=[{"role": "user", "content": conversation + "\n" + transcribed_text}],
        stream=True
    )
    
    # we will return the transcribed prompt concatenated with the LLM response
    retVal = "User: " + transcribed_text + "\n" + "Assistant: "
    
    # for each chunk of text returned from the LLM, print it and concatenate to return value
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            retVal = retVal + chunk.choices[0].delta.content
            
    return retVal

# start with empty conversation
try:
    conversation = ""
    while True:
        conversation = conversation + "\n" + record_and_prompt(conversation, stream)
        input("\nPress Enter to continue the conversation, CTRL+C to end.")
except KeyboardInterrupt:
    print("\nClosing audio stream...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    