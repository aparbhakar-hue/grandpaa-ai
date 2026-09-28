import runpod
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

def handler(job):
    return {"status": "Grandpaa Voice Worker Ready"}

runpod.serverless.start({"handler": handler})
