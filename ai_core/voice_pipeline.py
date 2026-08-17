"""
voice_pipeline.py

Reusable library for running three models SEQUENTIALLY on a Jetson Nano's GPU:
  1. Whisper (speech-to-text)
  2. SmolLM  (text generation)
  3. Tacotron2 + HiFi-GAN (text-to-speech)

Only ONE model ever lives on the GPU at a time. Each function loads its model,
runs inference, then explicitly frees GPU memory before returning -- this is
the load/unload pattern needed to fit sequentially on a 2GB board.

Import this as a library:
    from voice_pipeline import transcribe, generate_response, synthesize_speech, voice_qa_pipeline

--------------------------------------------------------------------------
READ THIS BEFORE RELYING ON IT ON YOUR ACTUAL BOARD
--------------------------------------------------------------------------
I could not execute or verify this code in the environment I'm running in --
no GPU, no ARM architecture, no JetPack, and no network access to
huggingface.co or pytorch hub from here. This is written correctly against
each library's documented API, but JetPack 4.x's very old software stack
(Ubuntu 18.04, Python 3.6, an old PyTorch build installed from NVIDIA's own
Jetson-specific wheel -- NOT plain `pip install torch`) means real
compatibility issues are likely. Run the self-test at the bottom of this
file FIRST, on the actual board, before building anything on top of it.

Setup notes:
    - Install PyTorch from NVIDIA's Jetson wheel for your exact JetPack
      version (search "PyTorch for Jetson" on NVIDIA's developer forums) --
      a normal `pip install torch` will not give you a CUDA-enabled build.
    - pip install openai-whisper transformers soundfile unidecode inflect
    - The Tacotron2/HiFi-GAN weights download once via torch.hub the first
      time synthesize_speech() runs -- that one-time setup needs internet.
      After that, everything runs fully offline.
--------------------------------------------------------------------------
"""

import gc
import torch
from contextlib import contextmanager


def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def gpu_model(loader_fn):
    """
    Loads whatever loader_fn() returns, yields it for use, then deletes it
    and clears GPU memory on exit -- guaranteeing only one model occupies
    the GPU at a time, regardless of what loader_fn returns (one object or
    a tuple of several).
    """
    model = loader_fn()
    try:
        yield model
    finally:
        del model
        _clear_gpu()


# ============================================================
# 1. WHISPER — speech to text
# ============================================================

def _load_whisper():
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("tiny", device=device)  # smallest official Whisper model (39M params)


def transcribe(audio_path: str) -> str:
    """Transcribe a wav file to text using Whisper 'tiny'. Loads, runs, unloads."""
    with gpu_model(_load_whisper) as model:
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result["text"].strip()


# ============================================================
# 2. SMOLLM — text generation
# ============================================================

# Smallest SmolLM variant, BASE model (not -Instruct) -- deliberately, per your
# request: no chat fine-tuning, no injected context, just the raw model.
MODEL_ID_SMOLLM = "HuggingFaceTB/SmolLM-135M"


def _load_smollm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_SMOLLM)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_SMOLLM,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # fp16 mainly buys you MEMORY headroom here, not speed -- the Nano's
        # Maxwell GPU has no tensor cores, but halving weight size still
        # matters a lot on a 2GB board.
    ).to(device)
    model.eval()
    return tokenizer, model


def generate_response(prompt: str, max_new_tokens: int = 60) -> str:
    """
    Generate a short, direct continuation from SmolLM's base model.

    Deliberately minimal by design:
      - no system prompt, no retrieval context, no fine-tuning
      - greedy decoding (do_sample=False) for determinism
      - repetition penalty to reduce looping/rambling
      - output cut at the first newline, since a base (non-instruct) model
        will keep generating unrelated text past a natural stopping point
    """
    with gpu_model(_load_smollm) as (tokenizer, model):
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    continuation = full_text[len(prompt):].strip()
    return continuation.split("\n")[0].strip()  # cut rambling continuations short


# ============================================================
# 3. TACOTRON2 + HIFI-GAN — text to speech
# ============================================================

def _load_tts():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tacotron2 = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tacotron2",
        model_math="fp16" if device == "cuda" else "fp32",
    ).to(device).eval()

    hifigan = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_hifigan",
    ).to(device).eval()

    utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")
    return tacotron2, hifigan, utils, device


def synthesize_speech(text: str, out_wav_path: str = "response.wav", sample_rate: int = 22050) -> str:
    """
    Text -> spoken .wav via Tacotron2 (text -> mel spectrogram) + HiFi-GAN
    (mel spectrogram -> waveform). Loads both onto the GPU, runs, unloads.
    """
    import soundfile as sf

    with gpu_model(_load_tts) as (tacotron2, hifigan, utils, device):
        sequences, lengths = utils.prepare_input_sequence([text])
        sequences, lengths = sequences.to(device), lengths.to(device)
        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences, lengths)
            audio = hifigan(mel).squeeze(1)
        audio_np = audio[0].detach().cpu().numpy()

    sf.write(out_wav_path, audio_np, sample_rate)
    return out_wav_path


def synthesize_speech_fallback(text: str, out_wav_path: str = "response.wav") -> str:
    """
    Non-neural fallback if Tacotron2+HiFi-GAN doesn't run cleanly on your
    JetPack version. Uses espeak via pyttsx3 -- CPU-only, near-zero memory,
    guaranteed to work, at the cost of sounding robotic rather than natural.
    pip install pyttsx3
    """
    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, out_wav_path)
    engine.runAndWait()
    return out_wav_path


# ============================================================
# Full pipeline convenience wrapper
# ============================================================

def voice_qa_pipeline(input_audio_path: str, out_wav_path: str = "response.wav", use_tts_fallback: bool = False) -> dict:
    """
    Full chain: audio question -> text (Whisper) -> response text (SmolLM)
    -> spoken response (Tacotron2+HiFi-GAN, or fallback). Only one model is
    ever resident on the GPU at a time.
    """
    question_text = transcribe(input_audio_path)
    response_text = generate_response(question_text)

    if use_tts_fallback:
        audio_path = synthesize_speech_fallback(response_text, out_wav_path)
    else:
        audio_path = synthesize_speech(response_text, out_wav_path)

    return {
        "question_text": question_text,
        "response_text": response_text,
        "response_audio_path": audio_path,
    }


# ============================================================
# Standalone compatibility test -- RUN THIS FIRST ON THE ACTUAL BOARD
# ============================================================

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    print("\n[1/3] Testing Whisper load/unload...")
    try:
        with gpu_model(_load_whisper):
            print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n[2/3] Testing SmolLM load/unload...")
    try:
        with gpu_model(_load_smollm):
            print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n[3/3] Testing Tacotron2+HiFi-GAN load/unload...")
    try:
        with gpu_model(_load_tts):
            print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  -> Consider synthesize_speech_fallback() instead (pyttsx3).")

    print("\nResolve any FAILED step above before building on top of this file.")
