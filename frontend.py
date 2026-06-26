import os
# Reduce CUDA fragmentation on tight GPUs (must be set before torch loads CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import io
import re
import time
import json
import queue
import threading

import torch
import whisper
import nltk
import requests
import configparser
import numpy as np
import scipy.io.wavfile as wavfile
import gradio as gr
import html
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse
from pydub import AudioSegment

from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from styletts2 import tts
from datasets import Dataset

nltk.download('punkt_tab')


# =====================================================================
# 0. Hugging Face Authentication (Login)
# =====================================================================
from huggingface_hub import login
from dotenv import load_dotenv
import subprocess

# Load the environment variables from the .env file
load_dotenv()

# Retrieve the token
hf_token = os.getenv("HF_TOKEN")

# Print it directly
print("Your HF_TOKEN is:", hf_token)

# Log into Hugging Face
if hf_token:
    #!hf auth login --token $hf_token
    subprocess.run(["hf", "auth", "login", "--token", hf_token], check=True)
    print("Successfully logged into Hugging Face!")
else:
    print("Error: HF_TOKEN not found in .env file.")



# =====================================================================
# 1. Configuration & Global State Management
# =====================================================================
HF_TOKEN = "hf_xxx"
HF_TOKEN = os.environ.get("HF_TOKEN", HF_TOKEN)
CALLSIGN = "Windracers 123"   # synced from the "My Callsign" UI field

# Set False once the pipeline is verified. When True: verbose console logging
# AND filtered/low-confidence transcriptions are surfaced in the UI log (tagged)
# so you can tell "quiet field" apart from "broken pipeline".
LIVE_DEBUG = True

# Single lock guarding ALL Whisper access (loading + inference) so the
# live background worker and the manual upload path never call .transcribe()
# on the same model object concurrently.
whisper_lock = threading.Lock()


class TelemetryData:
    def __init__(self):
        self.lock = threading.Lock()
        self.altitude = 4500
        self.heading = 270
        self.latitude = 40.4259
        self.longitude = -86.9081
        self.next_waypoint = "Boiler"
        self.airspeed = 120

    def update(self, alt, hdg, lat, lon, waypoint, speed):
        with self.lock:
            self.altitude = int(alt)
            self.heading = int(hdg)
            self.latitude = float(lat)
            self.longitude = float(lon)
            self.next_waypoint = str(waypoint).capitalize()
            self.airspeed = int(speed)

    def get_current_packet_string(self):
        with self.lock:
            return (f"[CURRENT AIRCRAFT TELEMETRY]:\n"
                    f"Altitude: {self.altitude} feet\n"
                    f"Heading: {self.heading} degrees\n"
                    f"Latitude: {self.latitude}\n"
                    f"Longitude: {self.longitude}\n"
                    f"Next Waypoint: {self.next_waypoint}\n"
                    f"Airspeed: {self.airspeed} knots")


telemetry = TelemetryData()


# Standard ICAO Word Translator
def to_icao_spelling(value):
    digit_map = {
        '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Tree', '4': 'Four',
        '5': 'Fife', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Niner',
        '.': 'Decimal', '-': 'Minus'
    }
    cleaned_string = str(value).upper()
    spelled_out = []
    for char in cleaned_string:
        if char in digit_map:
            spelled_out.append(digit_map[char])
        elif char.isalnum():
            spelled_out.append(char)
    return "-".join(spelled_out)


def generate_icao_telemetry_report():
    with telemetry.lock:
        alt_str = to_icao_spelling(telemetry.altitude)
        hdg_str = to_icao_spelling(telemetry.heading)
        lat_str = to_icao_spelling(telemetry.latitude).replace("-", " ")
        lon_str = to_icao_spelling(telemetry.longitude).replace("-", " ")
        spd_str = to_icao_spelling(telemetry.airspeed)
        way_str = telemetry.next_waypoint

    callsign_suffix = f" {clean_and_hyphenate_callsigns(CALLSIGN)}." if CALLSIGN.strip() else ""

    return (
        f"Altitude {alt_str}. "
        f"Heading {hdg_str}. "
        f"Latitude {lat_str}. "
        f"Longitude {lon_str}. "
        f"Next Waypoint {way_str}. "
        f"Airspeed {spd_str}."
        f"{callsign_suffix}"
    )


def clean_and_hyphenate_callsigns(text):
    digit_map = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Tree', '4': 'Four',
                 '5': 'Fife', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Niner'}

    def replacer(match):
        name = match.group(1).capitalize()
        digits = match.group(2)
        return f"{name}-" + "-".join([digit_map[d] for d in digits])

    pattern = r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(\d+)\b"
    return re.compile(pattern, re.IGNORECASE).sub(replacer, text)


PILOT_SYSTEM_PROMPT = f"""You are the pilot flying an aircraft with given callsign.
You are listening to an Air Traffic Controller (ATC) transmission.
You are equipped with a live telemetry feed showing your current aircraft instruments.

CRITICAL DISCIPLINE RULES:
1. NEVER hallucinate maneuvers. If ATC says "taxi into position and hold", do NOT say you are climbing or changing heading.
2. If ATC asks for a position or instrument report, read back the information ONLY from the provided [CURRENT AIRCRAFT TELEMETRY] packet.
3. Formulate all numerical values (digits) into distinct single words following ICAO radiotelephony rules:
   - 492 is "Four-Nine-Two". 9 is "Niner". 3 is "Tree". 5 is "Fife".
   - "Say "decimal" instead of "point" for decimal points e.g. 32.31 is "Tree-Two-Decimal-Three-One".
   - Say "minus" instead of "negative" for negative numbers e.g. -123 is "Minus-One-Two-Three".
   - For altitude, Say "Tousand" and "Hundred" instead of digits e.g. 12300 is "Twelve-Tousand-Three-Hundred".
   - for heading, Say each digits e.g. 270 is "Two-Seven-Zero".
4. Always end your transmission with the given official callsign.
5. Be highly concise. Do not use polite conversational filler.
"""

chat_history = [{"role": "system", "content": PILOT_SYSTEM_PROMPT}]


def reset_chat():
    # Backend conversation history
    chat_history.clear()
    chat_history.append({"role": "system", "content": PILOT_SYSTEM_PROMPT})
    # Drop any transcriptions already buffered in the queue, otherwise the next
    # timer tick would immediately repopulate the log we are about to clear.
    stream_manager._drain(stream_manager.results_queue)
    # Returned values clear, in order: live_log_state, live_log_display,
    # stt_output, llm_output. (Drop the last two if you'd rather keep the draft.)
    return "", "", "", ""


# Dynamic Model Trackers
loaded_whisper_size = None
whisper_model = None

loaded_llm_repo = None
llama_pipe = None


def get_whisper_model(size_name):
    """Shared, lock-guarded Whisper loader. Used by BOTH the manual upload
    path and the live-stream worker. Keep the live ASR size equal to the
    manual ASR size to avoid holding two models in memory at once."""
    global loaded_whisper_size, whisper_model
    with whisper_lock:
        if whisper_model is None or loaded_whisper_size != size_name:
            print(f"⏳ Dynamic Swapping: Loading Whisper Model Variant [{size_name}]...")
            whisper_model = whisper.load_model(size_name)
            loaded_whisper_size = size_name
    return whisper_model


def _load_llama_base():
    """Load Llama-3.1-8B at the HIGHEST precision the GPU can hold, degrading
    bf16 -> 8-bit -> 4-bit -> CPU-offload only as memory requires. A bigger GPU
    therefore gets full quality automatically; a small one still loads."""
    name = "meta-llama/Llama-3.1-8B-Instruct"
    if not torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float32, token=HF_TOKEN
        )

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU: {total_gb:.0f} GB total VRAM")

    def bnb(bits):
        from transformers import BitsAndBytesConfig
        if bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    # Build the ladder. bf16 needs ~16 GB of weights + headroom for
    # Whisper/StyleTTS2/activations, so only attempt it with clear room.
    attempts = []
    # Thresholds reserve ~6 GB for Whisper + StyleTTS2 + VAD + activations, which
    # share this GPU. 8-bit weights are ~9 GB, 4-bit ~5.5 GB, bf16 ~16 GB.
    if total_gb >= 22:
        attempts.append(("bf16 full precision", dict(torch_dtype=torch.bfloat16, device_map={"": 0})))
    if total_gb >= 18:
        attempts.append(("8-bit (near-lossless)", dict(device_map={"": 0}, quantization_config=bnb(8))))
    attempts.append(("4-bit NF4", dict(device_map={"": 0}, quantization_config=bnb(4))))
    # Last resort: split across CPU. Slow, but no merge => device hooks stay intact.
    attempts.append(("bf16 + CPU offload", dict(torch_dtype=torch.bfloat16, device_map="auto")))

    for label, kw in attempts:
        try:
            m = AutoModelForCausalLM.from_pretrained(name, token=HF_TOKEN, **kw)
            print(f"   ✅ Llama loaded at: {label}")
            return m
        except Exception as e:
            print(f"   ⚠️ {label} didn't fit/load ({type(e).__name__}); trying next...")
            torch.cuda.empty_cache()
    raise RuntimeError("Could not load Llama-3.1-8B in any configuration.")


def get_llm_pipeline(model_choice, custom_repo):
    global loaded_llm_repo, llama_pipe

    if model_choice == "Finetuned Model SCOPE 1.0":
        target_repo = "Sabine-Brunswicker/ATC-LLAMA-LORA"
        is_lora = True
    else:
        target_repo = "meta-llama/Llama-3.1-8B-Instruct"
        is_lora = False

    if llama_pipe is None or loaded_llm_repo != target_repo:
        print(f"⏳ Loading [{target_repo}]...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN
        )

        # Pick the best precision the current GPU can actually hold.
        base_model = _load_llama_base()

        if is_lora:
            # Keep the adapter ATTACHED — do NOT merge_and_unload(). Merging both
            # strips the accelerate device hooks (the original crash) and is not
            # possible on a 4-bit base. Inference output is identical, just unmerged.
            base_model = PeftModel.from_pretrained(base_model, target_repo, token=HF_TOKEN)

        llama_pipe = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
        )
        loaded_llm_repo = target_repo

    return llama_pipe


print("⏳ Initializing Default StyleTTS2 Vocal Engine...")
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load
try:
    styletts_engine = tts.StyleTTS2()
finally:
    torch.load = original_torch_load


# =====================================================================
# 2. Shared LLM inference (the convergence point for BOTH modes)
# =====================================================================
def set_callsign(cs):
    """Keep the global CALLSIGN in sync with the 'My Callsign' UI field so the
    ICAO readback and the LLM sign-off both use the current value."""
    global CALLSIGN
    CALLSIGN = (cs or "").strip()


def run_llm_inference(atc_text, llm_choice, custom_repo):
    """Takes already-transcribed ATC text and produces the pilot draft.
    Manual mode reaches this after a Whisper pass; live mode reaches it with
    text supplied by the streaming worker."""
    if not atc_text or not atc_text.strip():
        return atc_text, "⚠️ No ATC text available to respond to."
    try:
        active_llm = get_llm_pipeline(llm_choice, custom_repo)
        current_telemetry = telemetry.get_current_packet_string()
        callsign_line = f"[YOUR CALLSIGN]: {CALLSIGN}\n" if CALLSIGN.strip() else ""
        combined_payload = f"{callsign_line}{current_telemetry}\n\nATC Transmission: {atc_text}"

        active_turn = chat_history + [{"role": "user", "content": combined_payload}]
        formatted_prompt = active_llm.tokenizer.apply_chat_template(
            active_turn, tokenize=False, add_generation_prompt=True
        )
        outputs = active_llm(
            formatted_prompt,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )
        pilot_response = outputs[0]["generated_text"].strip()
        return atc_text, pilot_response
    except Exception as e:
        import traceback
        return atc_text, f"LLM Error: {str(e)}\n{traceback.format_exc()}"


# =====================================================================
# 3. Live ATC Stream Manager  (was: stream_and_transcribe_threaded)
# =====================================================================
def extract_stream_url(source):
    """Accepts a direct stream URL or a path to a .pls playlist file."""
    if not source:
        return None
    if isinstance(source, str) and source.lower().endswith(".pls") and os.path.exists(source):
        cfg = configparser.ConfigParser()
        try:
            cfg.read(source)
            return cfg.get('playlist', 'File1')
        except Exception:
            return None
    return source


class LiveStreamManager:
    """Owns the network-listener thread and the Whisper-worker thread.
    Finished, filtered ATC transcriptions are pushed onto results_queue,
    which the Gradio gr.Timer drains into the UI."""

    AVIATION_KEYWORDS = [
        "tower", "ground", "runway", "taxi", "cleared", "takeoff", "land",
        "hold", "short", "november", "cessna", "boiler", "archer", "skyhawk",
        "piper", "turn", "contact", "maintain", "altitude", "wind", "altimeter",
        "knots", "heading", "report", "traffic", "squawk", "approaching", "ready", "purdue"
    ]
    NUMBER_WORDS = ["one", "two", "three", "four", "five", "six",
                    "seven", "eight", "nine", "zero"]
    ATC_PROMPT = (
        "Lafayette Tower, Purdue, KLAF, Cessna, Boiler, Archer, November, Runway, Taxiway, "
        "cleared for takeoff, cleared to land, wind, altimeter, holding short, squawk, option."
    )

    def __init__(self):
        self.results_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._threads = []
        self.running = False
        self.model = None
        self.vad_model = None
        self.get_speech_timestamps = None
        self.callsign_filter = ""

    @staticmethod
    def _drain(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _ensure_vad(self):
        if self.vad_model is None:
            print("🧵 [Thread] Loading Silero VAD...")
            vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True
            )
            self.vad_model = vad_model
            self.get_speech_timestamps = utils[0]

    # ---- lifecycle ---------------------------------------------------
    def start(self, source, model_size, callsign_filter=""):
        if self.running:
            return "🟢 Live ATC stream is already running."

        stream_url = extract_stream_url(source)
        if not stream_url:
            return "❌ Could not resolve a stream URL. Paste a direct URL or upload a valid .pls file."

        self._stop_event.clear()
        self._drain(self._audio_queue)
        self._drain(self.results_queue)

        try:
            self.model = get_whisper_model(model_size)   # shared with manual path
            self._ensure_vad()
        except Exception as e:
            return f"❌ Failed to initialize ASR/VAD models: {e}"

        self.callsign_filter = (callsign_filter or "").strip()

        worker = threading.Thread(target=self._whisper_worker, daemon=True)
        listener = threading.Thread(target=self._network_listener, args=(stream_url,), daemon=True)
        self._threads = [worker, listener]
        worker.start()
        listener.start()
        self.running = True
        return f"🟢 Live ATC listening started ({model_size}). Source: {stream_url[:55]}..."

    def stop(self):
        if not self.running:
            return "⚪ Live ATC stream is not running."
        self._stop_event.set()
        self._audio_queue.put(None)   # wake the worker so it can exit
        self.running = False
        return "🛑 Live ATC listening stopped."

    # ---- worker: VAD-split -> per-transmission Whisper -> filter -----
    def _emit_transmission(self, text):
        """Guard ONE transmission and route it to the UI as a single request."""
        if not text:
            return
        text = text.strip()
        if len(text) <= 5:
            return
        low = text.lower()
        has_kw = any(k in low for k in self.AVIATION_KEYWORDS)
        has_num = any(c.isdigit() for c in low) or \
                  any(w in low for w in self.NUMBER_WORDS)
        passes = has_kw and has_num
        if self.callsign_filter:
            passes = passes and (self.callsign_filter.lower() in low)
        ts = time.strftime('%H:%M:%S')
        if passes:
            print(f"\n[{ts}] ✈️ ATC: \"{text}\"\n" + "-" * 60)
            self.results_queue.put((ts, text))
        elif LIVE_DEBUG:
            # Console-only: helps debugging but never reaches the UI log/request.
            reason = ("no keyword" if not has_kw else
                      "no number" if not has_num else "callsign filter")
            print(f"   (dropped [{reason}]): {text!r}")

    def _whisper_worker(self):
        print("🧵 [Thread] Whisper worker ready and waiting for speech...")
        while not self._stop_event.is_set():
            try:
                buf = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if buf is None:
                self._audio_queue.task_done()
                break
            try:
                wav_io = io.BytesIO()
                buf.set_frame_rate(16000).set_channels(1).export(wav_io, format="wav")
                wav_io.seek(0)
                seg = AudioSegment.from_wav(wav_io)
                samples = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
                dur = len(samples) / 16000.0

                # 1. Neural VAD -> split the buffer into individual transmissions.
                #    Brief within-sentence pauses (<500ms) stay grouped; a real
                #    push-to-talk gap starts a NEW segment. This is what stops one
                #    log line from containing several aircraft / instructions.
                speech_ts = self.get_speech_timestamps(
                    torch.from_numpy(samples), self.vad_model,
                    sampling_rate=16000, threshold=0.5,
                    min_silence_duration_ms=500,
                    min_speech_duration_ms=400,
                    speech_pad_ms=200,
                )
                if LIVE_DEBUG:
                    print(f"🎧 worker: {dur:.1f}s buffer -> {len(speech_ts)} transmission(s)")

                # 2. Transcribe EACH segment on its own: one transmission = one request.
                for st in speech_ts:
                    if self._stop_event.is_set():
                        break
                    a0, a1 = int(st['start']), int(st['end'])
                    seg_samples = samples[a0:a1]
                    if len(seg_samples) < int(0.3 * 16000):   # skip sub-0.3s blips
                        continue
                    with whisper_lock:
                        result = self.model.transcribe(
                            seg_samples, language="en", fp16=torch.cuda.is_available(),
                            initial_prompt=self.ATC_PROMPT,
                            no_speech_threshold=0.3, logprob_threshold=-0.6
                        )
                    text = result["text"].strip()
                    if LIVE_DEBUG:
                        print(f"📝 segment [{a0/16000:.1f}-{a1/16000:.1f}s]: {text!r}")
                    self._emit_transmission(text)
            except Exception as e:
                if LIVE_DEBUG:
                    import traceback
                    print("⚠️ worker error:", e)
                    traceback.print_exc()
            finally:
                self._audio_queue.task_done()
        print("🛑 [Thread] Whisper worker stopped.")

    # ---- listener: network stream -> phrase chunks -> audio_queue ----
    def _network_listener(self, stream_url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': '*/*', 'Icy-MetaData': '1'
        }
        SILENCE_THRESHOLD_DB = -27.0
        MIN_SILENCE_DURATION = 1.2
        MAX_TRANSMISSION_SECS = 12.0

        print(f"🟢 Live network listener active. Connecting to {stream_url}")
        while not self._stop_event.is_set():
            try:
                response = requests.get(stream_url, headers=headers, stream=True,
                                        allow_redirects=False, timeout=5)
                if LIVE_DEBUG:
                    print(f"🌐 upstream status: {response.status_code} | "
                          f"content-type: {response.headers.get('Content-Type')}")
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_url = response.headers['Location'].replace("https://", "http://")
                    if LIVE_DEBUG:
                        print(f"↪️ redirect -> {redirect_url}")
                    response = requests.get(redirect_url, headers=headers, stream=True, timeout=5)

                if response.status_code >= 400:
                    print(f"❌ upstream returned HTTP {response.status_code}; retrying in 3s")
                    response.close()
                    time.sleep(3)
                    continue

                print("🟢 Network feed connected. Receiving live chunks...")
                raw_buffer = b""
                phrase_buffer = AudioSegment.empty()
                silence_start = None
                is_speaking = False
                last_beat = time.time()
                recent_db = None

                for chunk in response.iter_content(chunk_size=4096):
                    if self._stop_event.is_set():
                        break
                    if not chunk:
                        continue
                    raw_buffer += chunk

                    if len(raw_buffer) >= 16384:
                        try:
                            seg = AudioSegment.from_file(io.BytesIO(raw_buffer), format="mp3")
                            raw_buffer = b""
                            phrase_buffer += seg
                            cur_db = seg.dBFS
                            recent_db = cur_db
                            cur_dur = len(phrase_buffer) / 1000.0

                            if cur_db > SILENCE_THRESHOLD_DB:
                                is_speaking = True
                                silence_start = None
                            else:
                                if is_speaking and silence_start is None:
                                    silence_start = time.time()

                            trig_silence = (is_speaking and silence_start and
                                            (time.time() - silence_start >= MIN_SILENCE_DURATION))
                            trig_timeout = (is_speaking and (cur_dur >= MAX_TRANSMISSION_SECS))

                            if trig_silence or trig_timeout:
                                if LIVE_DEBUG:
                                    print(f"➡️ queued phrase: {cur_dur:.1f}s "
                                          f"(trigger={'silence' if trig_silence else 'timeout'})")
                                self._audio_queue.put(phrase_buffer)
                                phrase_buffer = AudioSegment.empty()
                                is_speaking = False
                                silence_start = None
                        except Exception as e:
                            if LIVE_DEBUG and not isinstance(e, Exception):
                                pass  # decode hiccups on partial mp3 frames are normal
                            continue

                    # Heartbeat: proves bytes are flowing even when the field is silent.
                    if LIVE_DEBUG and (time.time() - last_beat) >= 5.0:
                        db_txt = f"{recent_db:.1f} dBFS" if recent_db is not None else "n/a"
                        gate = "above" if (recent_db is not None and recent_db > SILENCE_THRESHOLD_DB) else "below"
                        print(f"💓 receiving... last level {db_txt} ({gate} {SILENCE_THRESHOLD_DB} dB gate)")
                        last_beat = time.time()
                response.close()
            except Exception as e:
                if self._stop_event.is_set():
                    break
                if LIVE_DEBUG:
                    print("⚠️ listener error / reconnecting:", e)
                time.sleep(3)
                continue
        print("🛑 Live network listener stopped.")


stream_manager = LiveStreamManager()


# =====================================================================
# 4. Pipeline Control Processing Matrix
# =====================================================================
def sync_telemetry_inputs(alt, hdg, lat, lon, waypoint, speed):
    telemetry.update(alt, hdg, lat, lon, waypoint, speed)
    return telemetry.get_current_packet_string()


def generate_draft_response(audio_mic, audio_upload, asr_size, llm_choice, custom_llm_repo):
    """Manual mode: record/upload -> Whisper -> LLM."""
    active_audio_path = audio_upload if audio_upload is not None else audio_mic
    if active_audio_path is None:
        return "⚠️ Error: No audio recording or file upload detected.", ""

    try:
        active_whisper = get_whisper_model(asr_size)
        with whisper_lock:
            result = active_whisper.transcribe(
                active_audio_path,
                fp16=torch.cuda.is_available(),
                temperature=0.0,
                initial_prompt="Tower, Windracers, Runway, Taxi, Position, Hold, Line up, Cleared, Altimeter, Report Position"
            )
        atc_text = result["text"].strip()
        if not atc_text:
            return "❌ Speech Recognition failed to extract text.", ""

        return run_llm_inference(atc_text, llm_choice, custom_llm_repo)

    except Exception as e:
        import traceback
        return f"Draft Matrix Error: {str(e)}", traceback.format_exc()


def generate_from_transcript(atc_text, llm_choice, custom_llm_repo):
    """Live mode: text is already in the box -> LLM only (no Whisper)."""
    return run_llm_inference(atc_text, llm_choice, custom_llm_repo)


def approve_and_transmit_speech(atc_text, pilot_response):
    global chat_history
    if not pilot_response or "Error" in pilot_response or "failed" in pilot_response:
        return None, "⚠️ Cannot transmit an empty or faulty response draft.", ""

    try:
        chat_history.append({"role": "user", "content": atc_text})
        chat_history.append({"role": "assistant", "content": pilot_response})

        final_text = clean_and_hyphenate_callsigns(pilot_response).capitalize()
        sentences = nltk.sent_tokenize(final_text)
        combined_audio_chunks = []
        sample_rate = 24000
        temp_chunk_path = "temp_sentence_chunk.wav"

        for sentence in sentences:
            if not sentence.strip():
                continue
            styletts_engine.inference(text=sentence.strip(), output_wav_file=temp_chunk_path)
            sr, audio_data = wavfile.read(temp_chunk_path)
            sample_rate = sr
            combined_audio_chunks.append(audio_data)
            radio_pause = np.zeros(int(sample_rate * 0.2), dtype=audio_data.dtype)
            combined_audio_chunks.append(radio_pause)

        if os.path.exists(temp_chunk_path):
            os.remove(temp_chunk_path)

        final_audio_signal = np.concatenate(combined_audio_chunks)

        noise_level = 0.02
        white_noise = np.random.normal(0, noise_level, size=final_audio_signal.shape).astype(final_audio_signal.dtype)

        if final_audio_signal.dtype == np.float32:
            final_audio_signal = final_audio_signal + white_noise
            final_audio_signal = np.clip(final_audio_signal, -1.0, 1.0)
        else:
            max_val = np.iinfo(final_audio_signal.dtype).max
            noise_ints = (white_noise * max_val).astype(final_audio_signal.dtype)
            final_audio_signal = np.clip(
                final_audio_signal.astype(np.int32) + noise_ints.astype(np.int32),
                -max_val, max_val
            ).astype(final_audio_signal.dtype)

        # Unique filename per transmission so Gradio registers a value change and
        # autoplay re-fires every time (a fixed filename would only autoplay once).
        import glob
        for stale in glob.glob("pilot_transmission_*.wav"):
            try:
                os.remove(stale)
            except OSError:
                pass
        output_audio_path = f"pilot_transmission_{int(time.time() * 1000)}.wav"
        wavfile.write(output_audio_path, sample_rate, final_audio_signal)

        # Register as the active transmission and hand the player a controlled URL.
        _set_active_tx(output_audio_path)
        tx_url = f"/tx_audio?t={int(time.time() * 1000)}"
        return (output_audio_path,
                "✅ Radio Broadcast Transmission Complete with Static Filters Applied.",
                tx_url)

    except Exception as e:
        return None, f"StyleTTS2 Error: {str(e)}", ""


# ---- live-stream UI glue --------------------------------------------
def start_live_ui(url, pls_file, asr_size, callsign):
    # Textbox URL wins; .pls upload is only a fallback when the box is empty.
    source = url.strip() if (url and url.strip()) else pls_file
    # Filter the live feed on the operator word (first token of the callsign),
    # which is the most reliably transcribed part (e.g. "Windracers" from
    # "Windracers 123"). Blank callsign -> no filter (firehose).
    cs = (callsign or "").strip()
    callsign_filter = cs.split()[0] if cs else ""
    status = stream_manager.start(source, asr_size, callsign_filter)
    # Clear the log/state for a fresh session (stream_manager.start already
    # drained the queues). Returns: live_status, live_log_state, live_log_display.
    return status, "", ""


def stop_live_ui():
    return stream_manager.stop()


# ---- Same-origin audio proxy --------------------------------------------------
# Streams the active upstream feed through THIS server so the browser:
#   (a) connects same-origin/HTTPS  -> no http-stream-on-https mixed-content block
#   (b) the upstream fetch carries a browser User-Agent -> defeats 403 hotlink blocks
PROXY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*", "Icy-MetaData": "1",
}
_active_stream = {"url": None}
_active_stream_lock = threading.Lock()


def _set_active_stream(u):
    with _active_stream_lock:
        _active_stream["url"] = u


def _get_active_stream():
    with _active_stream_lock:
        return _active_stream["url"]


def atc_proxy(t: str = None):
    """FastAPI route: re-emits the currently active upstream stream. The `t`
    query arg is just a cache-buster so switching streams forces a reconnect."""
    stream_url = _get_active_stream()
    if not stream_url:
        return PlainTextResponse("No active stream. Click 'Load Stream Player' first.",
                                 status_code=404)
    try:
        upstream = requests.get(stream_url, headers=PROXY_HEADERS, stream=True,
                                allow_redirects=True, timeout=10)
    except Exception as e:
        return PlainTextResponse(f"Upstream connection error: {e}", status_code=502)
    if upstream.status_code >= 400:
        upstream.close()
        return PlainTextResponse(f"Upstream returned HTTP {upstream.status_code}",
                                 status_code=502)
    media_type = upstream.headers.get("Content-Type", "audio/mpeg")

    def gen():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        except Exception:
            pass
        finally:
            upstream.close()

    return StreamingResponse(gen(), media_type=media_type)


# ---- Transmission audio served through our own route -------------------------
# We control this URL, so playback doesn't depend on reading Gradio's audio DOM.
_active_tx = {"path": None}
_active_tx_lock = threading.Lock()


def _set_active_tx(p):
    with _active_tx_lock:
        _active_tx["path"] = p


def _get_active_tx():
    with _active_tx_lock:
        return _active_tx["path"]


def tx_audio(t: str = None):
    """FastAPI route: serves the most recent transmission WAV. The `t` arg is a
    cache-buster so each transmission is fetched fresh."""
    path = _get_active_tx()
    if not path or not os.path.exists(path):
        return PlainTextResponse("No transmission available yet.", status_code=404)
    return FileResponse(path, media_type="audio/wav")


def load_stream_player(url, pls_file):
    """Resolve the stream URL (handles .pls), register it as the active feed,
    and return a browser audio player that pulls through the same-origin proxy."""
    source = url.strip() if (url and url.strip()) else pls_file
    stream_url = extract_stream_url(source)
    if not stream_url:
        return "<p style='color:#b00;'>\u274c No valid stream URL or .pls provided.</p>"
    _set_active_stream(stream_url)
    src = f"/atc_proxy?t={int(time.time())}"   # cache-buster -> reconnect on switch
    shown = html.escape(stream_url, quote=True)
    return (
        f'<audio controls preload="none" style="width:100%;">'
        f'<source src="{src}" type="audio/mpeg">'
        f'<source src="{src}" type="audio/aac">'
        f'Your browser cannot play this stream.'
        f'</audio>'
        f'<p style="font-size:0.78em;color:#666;margin-top:4px;">Proxying: {shown}</p>'
    )


def poll_live(current_log, llm_choice, custom_repo, auto_gen):
    """Fired by gr.Timer. Drains new transcriptions into the log + the shared
    draft box, optionally auto-running the LLM on the most recent one."""
    no_change = (current_log, gr.update(), gr.update(), gr.update())
    if not stream_manager.running:
        return no_change

    items = []
    while True:
        try:
            items.append(stream_manager.results_queue.get_nowait())
        except queue.Empty:
            break
    if not items:
        return no_change

    new_lines = [f"[{ts}] ✈️ {txt}" for ts, txt in items]
    new_log = (current_log + "\n".join(new_lines) + "\n") if current_log else ("\n".join(new_lines) + "\n")
    latest_text = items[-1][1]

    if auto_gen:
        _, pilot = run_llm_inference(latest_text, llm_choice, custom_repo)
        return new_log, gr.update(value=new_log), gr.update(value=latest_text), gr.update(value=pilot)

    return new_log, gr.update(value=new_log), gr.update(value=latest_text), gr.update()


# =====================================================================
# 5. Gradio Core Layout Construction
# =====================================================================
# Client-side seamless playback: once the user has clicked anything, the document
# holds "sticky activation", so this timer-driven .play() is permitted. It watches
# the outbound monitor's <audio> and plays each new transmission the instant its
# src changes (unique filenames guarantee a fresh src every time).
# Runs once on page load: bless the hidden player on the first user gesture so
# later programmatic playback is allowed on every browser (incl. Safari).
_SEAMLESS_UNLOCK_JS = """
() => {
  const a = document.getElementById('tx_seamless');
  let primed = false;
  const prime = () => {
    if (primed || !a) return;
    const p = a.play();
    if (p && p.then) {
      p.then(() => { a.pause(); a.currentTime = 0; primed = true; }).catch(() => {});
    } else {
      primed = true;
    }
  };
  ['pointerdown', 'touchend', 'click', 'keydown'].forEach(
    ev => window.addEventListener(ev, prime, true)
  );
}
"""

# Runs right AFTER the transmit handler returns: read the controlled /tx_audio
# URL from the hidden field and play it through the blessed element.
_PLAY_TX_JS = """
(url) => {
  const a = document.getElementById('tx_seamless');
  if (!a) return;
  let u = (url || '').trim();
  if (!u) {
    const box = document.querySelector('#tx_url textarea, #tx_url input');
    u = box ? (box.value || '').trim() : '';
  }
  if (!u) return;
  a.src = u;
  a.currentTime = 0;
  const p = a.play();
  if (p && p.catch) p.catch(() => {});
}
"""


with gr.Blocks() as demo:   # theme moved to demo.launch() in Gradio 6.0
    gr.Markdown("# ✈️ Windracers UAV-ATC Communication Assistant")
    gr.Markdown("Live Stream **or** Manual Upload → Review the AI-generated Compliant Response → Transmit.")

    with gr.Row():

        # ── Left Column: Configuration & Telemetry ──────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🆔 Aircraft Identity")
            my_callsign_box = gr.Textbox(
                value="Windracers 123",
                label="My Callsign",
                info="",
            )
            my_callsign_box.change(fn=set_callsign, inputs=[my_callsign_box], outputs=None)

            gr.Markdown("### 🖨️ Active Telemetry Data")
            telemetry_display = gr.Code(
                value=telemetry.get_current_packet_string(),
                language="markdown",
                interactive=False,
            )

            gr.Markdown("### 📡 Live Avionics Overrides")
            slider_alt = gr.Slider(minimum=0, maximum=15000, step=100, value=4500, label="Altitude (ft)")
            slider_hdg = gr.Slider(minimum=0, maximum=360, step=1, value=270, label="Heading (deg)")
            input_lat = gr.Textbox(value="40.4259", label="Latitude")
            input_lon = gr.Textbox(value="-86.9081", label="Longitude")
            input_way = gr.Textbox(value="Boiler", label="Next Waypoint")
            slider_spd = gr.Slider(minimum=0, maximum=300, step=5, value=120, label="Airspeed (kts)")

            telemetry_inputs = [slider_alt, slider_hdg, input_lat, input_lon, input_way, slider_spd]
            for widget in telemetry_inputs:
                widget.change(
                    fn=sync_telemetry_inputs,
                    inputs=telemetry_inputs,
                    outputs=[telemetry_display],
                )

            reset_btn = gr.Button("🔄 Reset Chat", variant="secondary")

        # ── Right Column: Inputs (tabbed) + shared draft/transmit ────
        with gr.Column(scale=2):

            # Shared model config (used by both tabs)
            with gr.Row():
                dropdown_asr = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="medium",
                    label="Select ASR Model size",
                )
                dropdown_llm = gr.Dropdown(
                    choices=["Finetuned Model SCOPE 1.0", "Grammar-Prompted Model"],
                    value="Grammar-Prompted Model",
                    label="Select Language Model",
                )
            input_llm_repo = gr.State("Sabine-Brunswicker/ATC-LLAMA-LORA")

            with gr.Tabs():
                # ---- Tab A: Live ATC stream (default) -------------------
                with gr.Tab("📡 Live ATC Stream"):
                    with gr.Row():
                        stream_url_box = gr.Textbox(
                            label="ATC Stream URL (direct .mp3/.aac stream)",
                            value="http://d.liveatc.net/klaf",
                            placeholder="http://.../stream",
                            scale=3,
                        )
                        auto_gen_checkbox = gr.Checkbox(
                            label="Auto-generate draft for each transmission (only my callsign, if set)",
                            value=True,
                            scale=2,
                        )
                    with gr.Row():
                        listen_btn = gr.Button("🎧 Load / Refresh Stream Player", variant="secondary")
                    stream_player = gr.HTML(
                        value="<p style='color:#888;font-size:0.85em;'>No stream loaded.</p>",
                    )
                    # Hidden .pls fallback (resolved only if the URL box is empty).
                    pls_upload = gr.File(
                        label="…or upload a .pls playlist", file_types=[".pls"],
                        type="filepath", visible=False,
                    )
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Live Listening", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                    live_status = gr.Textbox(label="Live Status", value="⚪ Idle", interactive=False)
                    live_log_display = gr.Textbox(
                        label="📜 Live ATC Log", lines=10, interactive=False,
                        value="",
                    )
                    gen_from_live_btn = gr.Button("⚙️ Generate Response from latest transcript", variant="primary")

                # ---- Tab B: Manual audio --------------------------------
                with gr.Tab("🎙️ Manual Audio"):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record / Upload ATC Signal",
                    )
                    generate_btn = gr.Button("⚙️ Generate Response", variant="primary")

            # ── Shared draft + transmit panel (both modes write here) ──
            with gr.Group():
                stt_output = gr.Textbox(label="📻 Transcribed ATC Request", interactive=True)
                llm_output = gr.Textbox(label="✍️ Suggested UAV Response (Edit if necessary)", interactive=True)

            approve_btn = gr.Button("🛫 Authorize & Transmit Radio Signal", variant="primary")

            with gr.Group():
                transmission_status = gr.Textbox(
                    label="📡 Transmission Status Logs",
                    value="Awaiting Draft...",
                    interactive=False,
                )
                audio_output = gr.Audio(label="🔊 Outbound Broadcast Monitor", interactive=False, elem_id="tx_monitor")
                # Hidden, pre-unlocked player that produces the actual sound on every
                # browser (incl. Safari). Blessed by a silent play on first gesture.
                gr.HTML(
                    value='<audio id="tx_seamless" src="data:audio/wav;base64,UklGRgQCAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YeABAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIA=" preload="auto" style="display:none"></audio>'
                )
                # Hidden carrier: holds the /tx_audio URL for the seamless player.
                tx_url_box = gr.Textbox(visible=False, elem_id="tx_url")

    # Hidden state + timer for the live poller
    live_log_state = gr.State("")
    live_timer = gr.Timer(2.0)   # always active; poll_live no-ops when not streaming

    # ── Event Bindings ───────────────────────────────────────────────
    generate_btn.click(
        fn=generate_draft_response,
        inputs=[audio_input, audio_input, dropdown_asr, dropdown_llm, input_llm_repo],
        outputs=[stt_output, llm_output],
    )

    gen_from_live_btn.click(
        fn=generate_from_transcript,
        inputs=[stt_output, dropdown_llm, input_llm_repo],
        outputs=[stt_output, llm_output],
    )

    start_btn.click(
        fn=start_live_ui,
        inputs=[stream_url_box, pls_upload, dropdown_asr, my_callsign_box],
        outputs=[live_status, live_log_state, live_log_display],
    )
    stop_btn.click(fn=stop_live_ui, inputs=None, outputs=[live_status])

    listen_btn.click(
        fn=load_stream_player,
        inputs=[stream_url_box, pls_upload],
        outputs=[stream_player],
    )

    live_timer.tick(
        fn=poll_live,
        inputs=[live_log_state, dropdown_llm, input_llm_repo, auto_gen_checkbox],
        outputs=[live_log_state, live_log_display, stt_output, llm_output],
    )

    approve_btn.click(
        fn=approve_and_transmit_speech,
        inputs=[stt_output, llm_output],
        outputs=[audio_output, transmission_status, tx_url_box],
    ).then(fn=None, inputs=[tx_url_box], outputs=[], js=_PLAY_TX_JS)
    reset_btn.click(
        fn=reset_chat,
        inputs=None,
        outputs=[live_log_state, live_log_display, stt_output, llm_output],
    )

    # Bless the hidden player on first user gesture (enables Safari autoplay).
    demo.load(js=_SEAMLESS_UNLOCK_JS)

    # Reset the stream player on load so a persisted <audio autoplay> from a prior
    # session does not auto-start on reload; it only plays when Load is clicked.
    demo.load(
        fn=lambda: "<p style='color:#888;font-size:0.85em;'>No stream loaded.</p>",
        inputs=None, outputs=[stream_player],
    )


# ── Launch, then attach the audio proxy route to the SAME running app ─────────
# Shut down any server still running from a previous run of this cell so we
# don't stack ghost servers (daemon threads survive a cell stop on Colab).
gr.close_all()

app, local_url, share_url = demo.launch(
    theme=gr.themes.Soft(), share=True, prevent_thread_lock=True, inline=False
)

# Register custom routes AFTER launch and move them to the FRONT of the route
# table so Gradio's catch-all UI route doesn't shadow them.
app.add_api_route("/atc_proxy", atc_proxy, methods=["GET"])
app.add_api_route("/tx_audio", tx_audio, methods=["GET"])
app.router.routes.insert(0, app.router.routes.pop())   # /tx_audio to front
app.router.routes.insert(0, app.router.routes.pop())   # /atc_proxy to front

print(f"\U0001f50a Audio proxy attached. Local: {local_url}  Share: {share_url}")

# Keep the process alive. Stopping the cell (KeyboardInterrupt) now tears the
# server down so the /atc_proxy route closes and no audio keeps streaming.
try:
    demo.block_thread()
except KeyboardInterrupt:
    print("🛑 Shutting down: stopping listener and closing server...")
    try:
        stream_manager.stop()
    except Exception:
        pass
    demo.close()