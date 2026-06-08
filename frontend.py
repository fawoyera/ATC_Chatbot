import os
import torch
import whisper
import threading
import gradio as gr
from huggingface_hub import login, hf_hub_download 
from peft import PeftModel
from transformers import pipeline
from styletts2 import tts
import nltk
import re
import scipy.io.wavfile as wavfile
import numpy as np
nltk.download('punkt_tab')

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# 1. Configuration & Global State Management
# Load Hugging Face token from config.json
config_path = "./config.json"
try:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        hf_token = config.get("HF_ACCESS_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=True)
        print("Successfully logged into Hugging Face.")
    else:
        print("HF_ACCESS_TOKEN not found in config.json. Proceeding without authentication.")
except FileNotFoundError:
    print(f"config.json not found at {config_path}. Proceeding without Hugging Face authentication.")
except Exception as e:
    print(f"Error during Hugging Face login: {e}. Proceeding without authentication.")

HF_TOKEN = hf_token
#CALLSIGN = "Windracers One-Two-Tree"
CALLSIGN = ""

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

# Formats raw telemetry cleanly
def generate_icao_telemetry_report():
    with telemetry.lock:
        alt_str = to_icao_spelling(telemetry.altitude)
        hdg_str = to_icao_spelling(telemetry.heading)
        lat_str = to_icao_spelling(telemetry.latitude).replace("-", " ")
        lon_str = to_icao_spelling(telemetry.longitude).replace("-", " ")
        spd_str = to_icao_spelling(telemetry.airspeed)
        way_str = telemetry.next_waypoint

    # Only append callsign if it's set
    callsign_suffix = f" {to_icao_spelling(CALLSIGN)}." if CALLSIGN.strip() else ""

    return (
        f"Altitude {alt_str}. "
        f"Heading {hdg_str}. "
        f"Latitude {lat_str}. "
        f"Longitude {lon_str}. "
        f"Next Waypoint {way_str}. "
        f"Airspeed {spd_str}."
        f"{callsign_suffix}"
    )
    #return f"Altitude {alt_str}. Heading {hdg_str}. Latitude {lat_str}. Longitude {lon_str}. Next Waypoint {way_str}. Airspeed {spd_str}. {to_icao_spelling(CALLSIGN)}."

def clean_and_hyphenate_callsigns(text):
    digit_map = {'0':'Zero', '1':'One', '2':'Two', '3':'Tree', '4':'Four', '5':'Fife', '6':'Six', '7':'Seven', '8':'Eight', '9':'Niner'}
    def replacer(match):
        #name = match.group(1).upper()
        name = match.group(1).capitalize()
        digits = match.group(2)
        return f"{name}-" + "-".join([digit_map[d] for d in digits])
    pattern = r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(\d+)\b"
    return re.compile(pattern, re.IGNORECASE).sub(replacer, text)

#PILOT_SYSTEM_PROMPT = f"""You are the pilot flying an aircraft with callsign "{CALLSIGN}".
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

# Dynamic Model Trackers
loaded_whisper_size = None
whisper_model = None

loaded_llm_repo = None
llama_pipe = None

def get_whisper_model(size_name):
    global loaded_whisper_size, whisper_model
    if whisper_model is None or loaded_whisper_size != size_name:
        print(f"⏳ Dynamic Swapping: Loading Whisper Model Variant [{size_name}]...")
        whisper_model = whisper.load_model(size_name)
        loaded_whisper_size = size_name
    return whisper_model

'''def get_llm_pipeline(model_choice, custom_repo):
    global loaded_llm_repo, llama_pipe
    target_repo = "meta-llama/Llama-3.1-8B-Instruct" if model_choice == "Grammar-Prompted Model" else custom_repo

    if not target_repo or target_repo.strip() == "":
        raise ValueError("Custom Fine-tuned repository path cannot be empty.")

    if llama_pipe is None or loaded_llm_repo != target_repo:
        print(f"⏳ Dynamic Swapping: Initializing LLM Pipeline [{target_repo}]...")
        llama_pipe = pipeline(
            "text-generation",
            model=target_repo,
            dtype=torch.bfloat16,
            #device_map="auto",
            token=HF_TOKEN
        )
        loaded_llm_repo = target_repo
    return llama_pipe'''


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
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HF_TOKEN
        )
        if is_lora:
            base_model = PeftModel.from_pretrained(base_model, target_repo, token=HF_TOKEN)
            base_model = base_model.merge_and_unload()  # fuses adapter for faster inference

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


# 2. Pipeline Control Processing Matrix
def sync_telemetry_inputs(alt, hdg, lat, lon, waypoint, speed):
    telemetry.update(alt, hdg, lat, lon, waypoint, speed)
    return telemetry.get_current_packet_string()

def generate_draft_response(audio_mic, audio_upload, asr_size, llm_choice, custom_llm_repo):
    global chat_history

    # Select whichever audio slot contains data (Prioritize uploaded files)
    active_audio_path = audio_upload if audio_upload is not None else audio_mic
    if active_audio_path is None:
        return "⚠️ Error: No audio recording or file upload detected.", ""

    try:
        # Dynamic hardware validation & model activation
        active_whisper = get_whisper_model(asr_size)
        active_llm = get_llm_pipeline(llm_choice, custom_llm_repo)

        result = active_whisper.transcribe(
            active_audio_path,
            fp16=torch.cuda.is_available(),
            temperature=0.0,
            initial_prompt="Tower, Windracers, Runway, Taxi, Position, Hold, Line up, Cleared, Altimeter, Report Position"
        )
        atc_text = result["text"].strip()

        if not atc_text:
            return "❌ Speech Recognition failed to extract text.", ""

        '''if re.search(r"\b(report position|say position|report altitude|say status)\b", atc_text.lower()):
            pilot_response = generate_icao_telemetry_report()
            return atc_text, pilot_response'''

        current_telemetry = telemetry.get_current_packet_string()
        combined_payload = f"{current_telemetry}\n\nATC Transmission: {atc_text}"

        active_turn = chat_history + [{"role": "user", "content": combined_payload}]
        formatted_prompt = active_llm.tokenizer.apply_chat_template(active_turn, tokenize=False, add_generation_prompt=True)

        outputs = active_llm(
            formatted_prompt,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )

        pilot_response = outputs[0]["generated_text"].strip()
        pilot_response = clean_and_hyphenate_callsigns(pilot_response)
        return atc_text, pilot_response

    except Exception as e:
        import traceback
        return f"Draft Matrix Error: {str(e)}", traceback.format_exc()

def approve_and_transmit_speech(atc_text, pilot_response):
    global chat_history
    if not pilot_response or "Error" in pilot_response or "failed" in pilot_response:
        return None, "⚠️ Cannot transmit an empty or faulty response draft."

    print(f"DEBUG pilot_response: {repr(pilot_response)}")  
    print(f"DEBUG sentences: {nltk.sent_tokenize(pilot_response)}")  

    try:
        chat_history.append({"role": "user", "content": atc_text})
        chat_history.append({"role": "assistant", "content": pilot_response})

        final_text = clean_and_hyphenate_callsigns(pilot_response).capitalize()
        #final_text = pilot_response
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

        # Audio datatype aligned mix loop
        noise_level = 0.02
        white_noise = np.random.normal(0, noise_level, size=final_audio_signal.shape).astype(final_audio_signal.dtype)

        if final_audio_signal.dtype == np.float32:
            final_audio_signal = final_audio_signal + white_noise
            final_audio_signal = np.clip(final_audio_signal, -1.0, 1.0)
        else:
            max_val = np.iinfo(final_audio_signal.dtype).max
            noise_ints = (white_noise * max_val).astype(final_audio_signal.dtype)
            final_audio_signal = np.clip(final_audio_signal.astype(np.int32) + noise_ints.astype(np.int32), -max_val, max_val).astype(final_audio_signal.dtype)

        output_audio_path = "pilot_transmission.wav"
        wavfile.write(output_audio_path, sample_rate, final_audio_signal)

        status_update = "✅ Radio Broadcast Transmission Complete with Static Filters Applied."
        return output_audio_path, status_update

    except Exception as e:
        return None, f"StyleTTS2 Error: {str(e)}"

# 3. Gradio Core Layout Construction
with gr.Blocks(theme=gr.themes.Soft()) as demo:
#with gr.Blocks() as demo:
    gr.Markdown("# ✈️ Windracers UAV-ATC Communication Assistant")
    gr.Markdown("Stream, Record or Upload incoming ATC request. Review the AI-generated compliant response before transmitting.")

    with gr.Row():

        # ── Left Column: Configuration & Telemetry ──────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🖨️ Active Output Telemetry Data")
            telemetry_display = gr.Code(
                value=telemetry.get_current_packet_string(),
                language="markdown",
                interactive=False,
            )

            gr.Markdown("### 📡 Live Avionics Overrides")
            slider_alt = gr.Slider(minimum=0,   maximum=15000, step=100, value=4500,  label="Altitude (ft)")
            slider_hdg = gr.Slider(minimum=0,   maximum=360,   step=1,   value=270,   label="Heading (deg)")
            input_lat  = gr.Textbox(value="40.4259",  label="Latitude")
            input_lon  = gr.Textbox(value="-86.9081", label="Longitude")
            input_way  = gr.Textbox(value="Boiler",   label="Next Waypoint")
            slider_spd = gr.Slider(minimum=0,   maximum=300,   step=5,   value=120,   label="Airspeed (kts)")

            # Wire every telemetry slider/textbox to live-sync the display
            telemetry_inputs = [slider_alt, slider_hdg, input_lat, input_lon, input_way, slider_spd]
            for widget in telemetry_inputs:
                widget.change(
                    fn=sync_telemetry_inputs,
                    inputs=telemetry_inputs,
                    outputs=[telemetry_display],
                )

        # ── Right Column: Audio I/O & Pipeline ──────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📻 Input Audio Signal Acquisition")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record / Upload ATC Signal",
            )

            with gr.Row():
                dropdown_asr = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="medium",
                    label="Select ASR Model size",
                )
                dropdown_llm = gr.Dropdown(
                    choices=["Finetuned Model SCOPE 1.0", "Grammar-Prompted Model"],
                    value="Finetuned Model SCOPE 1.0",
                    label="Select Language Model",
                )

            #input_llm_repo = gr.State("meta-llama/Llama-3.1-8B-Instruct")
            input_llm_repo = gr.State("Sabine-Brunswicker/ATC-LLAMA-LORA")
            #input_llm_repo = gr.Textbox(
             #   value="meta-llama/Llama-3.1-8B-Instruct",
              #  label="SCOPE Huggingface Repo (Used for Fine-Tuned choice)",
            #)

            generate_btn = gr.Button("⚙️ Generate Response", variant="primary")

            with gr.Group():
                stt_output = gr.Textbox(label="📻 Transcribed ATC Request",      interactive=False)
                llm_output = gr.Textbox(label="✍️ Suggested UAV Response (Edit if necessary)", interactive=True)

            approve_btn = gr.Button("🛫 Authorize & Transmit Radio Signal", variant="primary")

            with gr.Group():
                transmission_status = gr.Textbox(
                    label="📡 Transmission Status Logs",
                    value="Awaiting Draft...",
                    interactive=False,
                )
                audio_output = gr.Audio(label="🔊 Outbound Broadcast Monitor", interactive=False)

    # ── Event Bindings ───────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_draft_response,
        inputs=[audio_input, audio_input, dropdown_asr, dropdown_llm, input_llm_repo],
        outputs=[stt_output, llm_output],
    )
    approve_btn.click(
        fn=approve_and_transmit_speech,
        inputs=[stt_output, llm_output],
        outputs=[audio_output, transmission_status],
    )

demo.launch(share=True, debug=True)