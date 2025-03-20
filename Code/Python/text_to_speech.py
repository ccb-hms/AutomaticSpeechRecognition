# An example of a transformers text-to-speech model to convert string data
# into audio waveform.

import torch
import numpy as np
import sounddevice as sd
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, pipeline
from datasets import load_dataset

# figure out what we have available for compute
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# set up the SpeechT5 pipeline
# on Apple Silicon, setting device to "mps" still runs the model on CPU, single threaded.
# explicitly setting to CPU at least lets it utilize multiple cores.
synthesizer = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)

# these are all of the voices available
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# provide a string in text, and an integer in voice_index to select a voice.
def text_to_speech(text, voice_index):
    
    # change this embedding to change the voice
    speaker_embedding = torch.tensor(embeddings_dataset[voice_index]["xvector"]).unsqueeze(0)
    
    speech = synthesizer(text, forward_params={"speaker_embeddings": speaker_embedding})
    sd.play(speech["audio"], speech["sampling_rate"])
    sd.wait()

# Example usage
text_to_speech("Testing text to speech!", 7306)