import os
import threading
import torch
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS

SAMPLE_RATE = 16000
CHANNELS = 1

def setup_whisper():
    print("Setiing Whisper... [1/3]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/whisper-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    return processor, model, device

def setup_smollm():
    print("Setiing SmolLM... [2/3]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tokenizer, model

def setup_tts():
    print("Setiing Tacotron + HiFi GAN... [3/3]")
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    tts = TTS(model_name=model_name, progress_bar=False)
    if torch.cuda.is_available():
        tts.to("cuda")
    return tts

def _wait_for_key(prompt, expected):
    #Block the system if the flag of the app is not triggered 
    while True:
        typed = input(prompt).strip().lower()
        if typed == expected:
            return
        print(f"[!] Type '{expected}' and press Enter.")

def record_user_voice():
    _wait_for_key("\nWaiting for the flag", "1")
    print("Recording")

    audio_buffer = []
    stop_event = threading.Event()
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms frames

    def _record_loop():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32') as stream:
            while not stop_event.is_set():
                chunk, _ = stream.read(chunk_size)
                audio_buffer.append(chunk)

    rec_thread = threading.Thread(target=_record_loop, daemon=True)
    rec_thread.start()

    #Block until condition 
    #TODO: Implement sielnce stop or push stop
    _wait_for_key("", "1")
    stop_event.set()
    rec_thread.join()

    print("Recording stopped. Thinking...")

    if not audio_buffer:
        return np.zeros((0, CHANNELS), dtype='float32')
    return np.concatenate(audio_buffer, axis=0)

def transcribe(audio_np, processor, model, device):
    input_features = processor(audio_np.squeeze(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

def generate_llm_response(user_text, tokenizer, model, chat_history):
    # Append new statement to rolling history array
    chat_history.append({"role": "user", "content": user_text})

    # Format tokens using chat template structure rules
    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the newly generated text block from response array
    new_tokens = outputs[0][inputs.shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Track the system's own answer back into history memory
    chat_history.append({"role": "assistant", "content": response_text})
    return response_text

def main():
    print("Andy AI Version 0.1")

    # Core system boot setup
    w_proc, w_model, device = setup_whisper()
    llm_tok, llm_model = setup_smollm()
    tts_engine = setup_tts()

    # Track long-term context memory in active runtime list
    chat_history = [
        {"role": "system", "content": "You are a helpful, friendly, and concise voice assistant about music. Give brief, direct answers suited for spoken responses."}
    ]

    temp_in = "temp_voice_in.wav"
    temp_out = "temp_voice_out.wav"
    print("\nSystem booted. Starting Pipeline")

    try:
        while True:
            #Capture audio
            audio_data = record_user_voice()
            if audio_data.shape[0] == 0:
                print("[!] No audio captured, try again.")
                continue

            wavfile.write(temp_in, SAMPLE_RATE, audio_data)

            #Extract text
            user_statement = transcribe(audio_data, w_proc, w_model, device)
            if len(user_statement) < 2:
                continue

            print(f"\n[You]: {user_statement}")

            #Terminal Commands, for a simple hands free exit 
            if any(word in user_statement.lower() for word in ["exit", "stop conversation", "goodbye"]):
                print("[Assistant]: Goodbye!")
                break

            #Prompt to generate text
            ai_response = generate_llm_response(user_statement, llm_tok, llm_model, chat_history)
            print(ai_response)

            #Generate speech
            tts_engine.tts_to_file(text=ai_response, file_path=temp_out)

            #Play audio
            fs, speaker_data = wavfile.read(temp_out)
            sd.play(speaker_data, fs)
            sd.wait()  # Block until hardware audio buffer completes rendering

    except KeyboardInterrupt:
        print("\nOur session ends here!")
    finally:
        for temp_file in [temp_in, temp_out]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    main()