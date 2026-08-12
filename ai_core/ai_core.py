"""
Consolidated AI core and server in one file.
Run: python ai_core.py

This file initializes model components (Whisper, SmolLM, TTS), exposes a small Flask
API used by the chat UI (chat.html), and provides utility functions to record on the
server microphone, transcribe, generate LLM responses, and render TTS.

Notes:
- Recording uses the server's audio device via sounddevice. For browser-based capture,
  a separate flow that uploads recorded audio is recommended.
- Model downloads may be large and require time/disk/VRAM.
"""

import os
import threading
import time
from typing import Optional, List, Dict

from flask import Flask, request, jsonify, send_from_directory, abort

# scientific / ML libs
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS

# -------------------- Configuration --------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_RECORD_SECONDS = 5
TTS_OUT_FILE = "temp_voice_out.wav"
TTS_IN_FILE = "temp_voice_in.wav"
HOST = '0.0.0.0'
PORT = 7860

# -------------------- Model handles (globals) --------------------
_w_proc = None
_w_model = None
_device = None
_llm_tok = None
_llm_model = None
_tts_engine = None

# -------------------- Setup helpers (from original main.py) --------------------

def setup_whisper():
    print("Setting up Whisper... [1/3]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/whisper-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    return processor, model, device


def setup_smollm():
    print("Setting up SmolLM... [2/3]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tokenizer, model


def setup_tts():
    print("Setting up TTS engine... [3/3]")
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    tts = TTS(model_name=model_name, progress_bar=False)
    if torch.cuda.is_available():
        try:
            tts.to("cuda")
        except Exception:
            # some TTS backends do not support .to() — ignore if not supported
            pass
    return tts

# -------------------- Core functions --------------------

def init_models():
    """Initialize or return existing model handles."""
    global _w_proc, _w_model, _device, _llm_tok, _llm_model, _tts_engine
    if _w_proc is None or _w_model is None or _device is None:
        _w_proc, _w_model, _device = setup_whisper()
    if _llm_tok is None or _llm_model is None:
        _llm_tok, _llm_model = setup_smollm()
    if _tts_engine is None:
        _tts_engine = setup_tts()
    return _w_proc, _w_model, _device, _llm_tok, _llm_model, _tts_engine


def transcribe(audio_np, processor, model, device):
    # audio_np expected shape: (N, channels) or (N,) ; use processor to handle
    if audio_np.ndim > 1:
        arr = audio_np.squeeze()
    else:
        arr = audio_np
    input_features = processor(arr, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def generate_llm_response(user_text, tokenizer, model, chat_history):
    # Build a local prompt history (don't mutate the real chat_history)
    prompt_history = [dict(m) for m in chat_history]
    prompt_history.append({"role": "user", "content": user_text})

    # Format tokens using chat template structure rules
    # Some tokenizers may not have apply_chat_template — guard for that
    if hasattr(tokenizer, 'apply_chat_template'):
        raw_inputs = tokenizer.apply_chat_template(
            prompt_history,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    else:
        # Fallback: build a simple prompt from conversation turns (exclude system messages entirely)
        conv_lines = []
        for m in prompt_history:
            if m.get('role') == 'system':
                continue
            # Keep raw content without role-label injection to preserve natural generation
            conv_lines.append(m.get('content', ''))
        if conv_lines:
            # Use prior lines followed by the latest user_text
            prompt = "\n".join(conv_lines)
        else:
            prompt = user_text
        raw_inputs = tokenizer(prompt, return_tensors='pt')

    # Ensure tensors are on the model device
    if isinstance(raw_inputs, dict):
        inputs = {k: v.to(model.device) for k, v in raw_inputs.items()}
    else:
        # Some tokenizers may return BatchEncoding which behaves like a dict
        try:
            inputs = {k: v.to(model.device) for k, v in raw_inputs.items()}
        except Exception:
            # last resort: try to move input_ids only
            inputs = raw_inputs
            if hasattr(inputs, 'input_ids'):
                try:
                    inputs['input_ids'] = inputs['input_ids'].to(model.device)
                except Exception:
                    pass

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        if isinstance(inputs, dict) and 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            else:
                outputs = model.generate(input_ids=input_ids, **gen_kwargs)
        else:
            try:
                outputs = model.generate(**inputs, **gen_kwargs)
            except Exception:
                # Last resort: encode the user_text only
                fallback_inputs = tokenizer(user_text, return_tensors='pt').to(model.device)
                outputs = model.generate(input_ids=fallback_inputs['input_ids'], **gen_kwargs)

    out_ids = outputs if not isinstance(outputs, (list, tuple)) else outputs[0]

    # Determine how many tokens were input so we can extract newly generated tokens
    try:
        if isinstance(inputs, dict) and 'input_ids' in inputs:
            input_len = inputs['input_ids'].shape[1]
        else:
            input_len = 0
    except Exception:
        input_len = 0

    try:
        gen_ids = out_ids[0] if out_ids.dim() == 2 else out_ids
        if input_len > 0 and gen_ids.shape[0] > input_len:
            new_tokens = gen_ids[input_len:]
        else:
            new_tokens = gen_ids
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception:
        try:
            raw_response = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        except Exception:
            raw_response = "(no response)"

    # Minimal cleanup only: remove obvious 'Response:' markers and role labels, but do not inject or keep system context
    import re
    text = raw_response.strip()

    # Normalize and strip 'Response:' style echoes
    text = re.sub(r"\*+Response:\*+", "Response:", text, flags=re.I)
    text = text.replace('[Response:', 'Response:')
    if re.search(r'(?i)response\s*:', text):
        parts = re.split(r'(?i)response\s*:', text)
        text = parts[-1].strip()

    # If the model included role prefixes like 'assistant:' or 'avatar:', take after the last label
    labels = list(re.finditer(r"(?i)(assistant:|avatar:|andy:)", text))
    if labels:
        last = labels[-1]
        text = text[last.end():].strip()

    # Remove leading 'user:' or 'system:' if present
    text = re.sub(r'^(user:|system:)\s*', '', text, flags=re.I).strip()

    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    text = re.sub(r"\s+", " ", text).strip()
    response_text = text if text else "(no response)"

    # Stateless: do not modify chat_history
    return response_text

# -------------------- High-level APIs --------------------

def run_voice_once(duration=DEFAULT_RECORD_SECONDS, chat_history: Optional[List[Dict]] = None) -> Dict:
    """Record on the server mic for duration seconds, transcribe, produce a reply and TTS.
    Returns: { transcription, ai_response, tts_file }
    """
    init_models()
    temp_in = TTS_IN_FILE
    temp_out = TTS_OUT_FILE

    # Record
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    audio_np = np.array(recording)

    # Normalize/write as int16 WAV
    wavfile.write(temp_in, SAMPLE_RATE, (audio_np * 32767).astype('int16'))

    transcription = transcribe(audio_np, _w_proc, _w_model, _device)

    if chat_history is None:
        chat_history = []
    ai_response = generate_llm_response(transcription, _llm_tok, _llm_model, chat_history)
    # Generate TTS
    _tts_engine.tts_to_file(text=ai_response, file_path=temp_out)

    return {"transcription": transcription, "ai_response": ai_response, "tts_file": temp_out}


def generate_text_reply(user_text: str, chat_history: Optional[List[Dict]] = None) -> str:
    init_models()
    if chat_history is None:
        chat_history = []
    return generate_llm_response(user_text, _llm_tok, _llm_model, chat_history)

# -------------------- Flask Server --------------------
app = Flask(__name__, static_folder='.')

# In-memory session store
session_histories: Dict[str, List[Dict]] = {}


def get_session_history(session_id: str):
    # Stateless sessions: always return an empty history so each request is independent.
    # This ensures no system instructions or past conversation are sent to the model.
    return []


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/chat')
def chat_page():
    return send_from_directory('.', 'chat.html')


@app.route('/api/send_message', methods=['POST'])
def api_send_message():
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    if not message:
        return jsonify({'error': 'empty message'}), 400
    chat_history = get_session_history(session_id)
    reply = generate_text_reply(message, chat_history)
    # Stateless: do not persist chat_history; return a single-turn reply
    return jsonify({'reply': reply, 'session_id': session_id})


@app.route('/api/run_main', methods=['POST'])
def api_run_main():
    data = request.get_json() or {}
    duration = int(data.get('duration', DEFAULT_RECORD_SECONDS))
    session_id = data.get('session_id', 'default')
    chat_history = get_session_history(session_id)
    try:
        result = run_voice_once(duration=duration, chat_history=chat_history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # Stateless: do not persist chat_history; return single-turn transcription and response
    return jsonify({
        'transcription': result['transcription'],
        'ai_response': result['ai_response'],
        'tts_file': result['tts_file'],
        'session_id': session_id
    })


# Simplified compatibility endpoint: POST /run_main
@app.route('/run_main', methods=['POST'])
def run_main_compat():
    """Backward-compatible endpoint that performs the same action as /api/run_main."""
    return api_run_main()


@app.route('/tts/<path:filename>')
def tts_file(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return abort(404)


# Generic static file serving (avoid catching API calls)
@app.route('/<path:filename>')
def serve_file(filename):
    if filename.startswith('api/') or filename.startswith('tts/'):
        return abort(404)
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return abort(404)


def preload_models_background():
    try:
        init_models()
    except Exception as e:
        print('Model preload failed:', e)


if __name__ == '__main__':
    # Start preload in background
    threading.Thread(target=preload_models_background, daemon=True).start()
    print(f"Starting server on http://{HOST}:{PORT} ...")
    app.run(host=HOST, port=PORT, debug=True)
