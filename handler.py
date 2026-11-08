import base64
import io
import os
import tempfile
import threading

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from scipy.io import wavfile

app = FastAPI(title="Grandpaa Voice Studio")
LOCK = threading.Lock()
CACHE = {}

MMS_LANGUAGES = {
    "pa": "facebook/mms-tts-pan",
    "hi-basic": "facebook/mms-tts-hin",
}
CHATTERBOX_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh"
}

def synthesize(text, language, engine, reference_audio=None, exaggeration=0.5):
    with LOCK:
        if engine == "mms":
            if language not in MMS_LANGUAGES:
                raise ValueError("MMS language not configured")
            from transformers import AutoTokenizer, VitsModel
            key = "mms:" + language
            if key not in CACHE:
                name = MMS_LANGUAGES[language]
                model = VitsModel.from_pretrained(name)
                model.eval()
                CACHE[key] = (AutoTokenizer.from_pretrained(name), model)
            tokenizer, model = CACHE[key]
            with torch.inference_mode():
                audio = model(**tokenizer(text, return_tensors="pt")).waveform.squeeze().cpu().numpy()
            rate = model.config.sampling_rate
        elif engine == "chatterbox":
            if language not in CHATTERBOX_LANGUAGES:
                raise ValueError("Chatterbox language not supported")
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            if "chatterbox" not in CACHE:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                CACHE["chatterbox"] = ChatterboxMultilingualTTS.from_pretrained(device=device)
            model = CACHE["chatterbox"]
            path = None
            try:
                if reference_audio:
                    raw = base64.b64decode(reference_audio, validate=True)
                    if len(raw) > 5_000_000:
                        raise ValueError("Reference audio exceeds 5 MB")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(raw)
                        path = f.name
                args = {"language_id": language, "exaggeration": float(exaggeration)}
                if path:
                    args["audio_prompt_path"] = path
                with torch.inference_mode():
                    audio = model.generate(text, **args).squeeze().detach().cpu().numpy()
                rate = model.sr
            finally:
                if path and os.path.exists(path):
                    os.unlink(path)
        else:
            raise ValueError("Unknown engine")
    buffer = io.BytesIO()
    wavfile.write(buffer, rate, np.asarray(audio, dtype=np.float32))
    return {"status": "completed", "language": language, "engine": engine,
            "format": "wav", "sample_rate": rate,
            "audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}

@app.get("/health")
def health():
    return {"status": "ok", "service": "grandpaa-voice-studio",
            "engines": ["chatterbox", "mms"], "gpu_available": torch.cuda.is_available()}

@app.post("/run")
async def run(request: Request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        item = data.get("input", data)
        if not isinstance(item, dict):
            raise ValueError("Expected input object")
        if item.get("service", "test") == "test":
            return {"output": {"status": "Grandpaa GridShare connection OK", "received": item}}
        if item.get("service") != "voice":
            raise ValueError("Unsupported service")
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("Text must be 1-500 characters")
        language = item.get("language", "hi")
        engine = item.get("engine", "mms" if language == "pa" else "chatterbox")
        exaggeration = item.get("exaggeration", 0.5)
        if not isinstance(exaggeration, (int, float)) or not 0 <= exaggeration <= 1:
            raise ValueError("exaggeration must be between 0 and 1")
        result = synthesize(text.strip(), language, engine,
                            item.get("reference_audio_base64"), exaggeration)
        return {"output": result}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        print("Voice worker failure:", type(exc).__name__, str(exc))
        return JSONResponse(status_code=500, content={"error": "Voice generation failed"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
