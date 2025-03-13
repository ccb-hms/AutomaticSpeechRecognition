# Minimal example to run OpenAI's Whisper Large v3 automatic speech recognition model.
# 
# Here are thre prerequisites when running on a DGX on the Longwood cluster:
#
# conda create -n asr python=3.12.7
# conda activate asr
#
# For Apple Silicon MPS devices, use this instead of conda install pytorch... below:
# pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
#
# conda install transformers
# conda install -c conda-forge accelerate
# conda install -c huggingface -c conda-forge datasets
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install -c conda-forge librosa
#
# module load dgx
# module load cuda/12.6
#
# huggingface-cli login

import torch
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

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

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

audio_input = {"raw": sample["array"], "sampling_rate": sample["sampling_rate"]}

# time it:
start_time = time.time()
result = pipe(audio_input, return_timestamps=True)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

# an example of transcribing from a file (accepts mp3 or flac only)
# for i in range(60):
#     result = pipe("/home/npp10/LoRA_abstract.flac", return_timestamps=True)

# time it:
# start_time = time.time()
# result = pipe("/home/npp10/LoRA_abstract.flac", return_timestamps=True)
# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Execution time: {elapsed_time:.4f} seconds")

print(result["text"])
