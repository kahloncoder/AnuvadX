import streamlit as st
import groq
import os
import io
import base64
from PIL import Image
import PyPDF2
import speech_recognition as sr
import threading
import cv2
import time
import requests
from gtts import gTTS
import tempfile
from langdetect import detect

# --- Constants ---
GROQ_CHAT_MODEL = "llama3-70b-8192"
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"
MAX_TEXT_PROCESSING_CHARS = 8000

# --- API Client Initialization ---

@st.cache_resource 
def get_groq_client():
    """Initializes and returns the Groq client, caching the resource."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("WARN: Groq API Key not found in environment.") 
        return None
    try:
        client = groq.Client(api_key=groq_key)
        print("INFO: Groq client initialized successfully.")
        # Basic check (optional)
        # client.models.list()
        return client
    except Exception as e:
        # Use st.error here ONLY if unavoidable, prefer returning errors
        st.sidebar.error(f"Failed to initialize Groq client: {e}")
        print(f"ERROR: Failed to initialize Groq client: {e}")
        return None

# --- Helper Functions ---

def detect_language(text):
    """Detects the language of the provided text."""
    if not text or not isinstance(text, str):
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en" 

# --- Core Logic: Text Processing & API Calls ---

def get_response_in_language(client, user_input, preferred_language, chat_history):
    """Gets a conversational response from Groq LLM in the preferred language."""
    if not client:
        return "Error: Groq API client not available."

    lang_map = {
        "simplified_english": "simplified English using basic vocabulary and simple sentence structures.",
        "hindi": "Hindi language using Devanagari script.",
        "hinglish": "Hinglish (mix of Hindi and English, with Hindi words written in Roman script).",
        "punjabi": "Punjabi language using Gurmukhi script.",
    }
    system_prompt = f"You are a helpful assistant. Always respond in {lang_map.get(preferred_language, 'English')}."

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history) # Use passed history
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=GROQ_CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error (Chat): {e}") # Keep st.error for visible feedback
        return f"Error generating chat response: {e}"

def get_word_for_word_translation(client, text, preferred_language):
    """Gets a word-for-word translation and summary from Groq LLM."""
    if not client: return "Error: Groq API client not available."
    if not text or not text.strip(): return "No text provided for translation."

    lang_map = {
        "simplified_english": "simplified English", "hindi": "Hindi using Devanagari script",
        "hinglish": "Hinglish (mix of Hindi and English with Hindi words in Roman script)",
        "punjabi": "Punjabi using Gurmukhi script"
    }
    target_lang_desc = lang_map.get(preferred_language, preferred_language.replace('_', ' '))
    system_message = f"You are a helpful translation assistant. Provide a word-for-word translation into {target_lang_desc}, followed by a brief summary of its meaning in the same target language."
    user_prompt = f"""Translate the following text word-for-word into {target_lang_desc}:

"{text[:MAX_TEXT_PROCESSING_CHARS]}" {"[...Text Truncated...]" if len(text) > MAX_TEXT_PROCESSING_CHARS else ""}

Format your response precisely as follows:
1.  **Original Text:** (Repeat the original text, truncated if necessary)
2.  **Word-for-Word Translation:** (Provide the word-by-word translation)
3.  **Summary Meaning:** (Provide a brief summary of the meaning in the target language)
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}],
            temperature=0.5, max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error (Word Translation): {e}")
        return f"Error during word translation: {e}"

def extract_text_from_pdf_basic(pdf_file_bytes):
    """Extracts text from PDF bytes using PyPDF2."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file_bytes))
        if not pdf_reader.pages:
             return "[Error: PDF seems empty or corrupted]"
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                text += page_text + "\n\n" if page_text else f"[Page {i+1} Blank or Unreadable]\n\n"
            except Exception as page_err:
                 print(f"WARN: Error extracting text from page {i+1}: {page_err}")
                 text += f"[Page {i+1} Text Extraction Error]\n\n"
        return text.strip() if text.strip() else "[No text extracted from PDF]"
    except Exception as e:
        print(f"ERROR: Failed extracting text from PDF: {e}")
        return f"[Error extracting text: {e}]"

def summarize_text_in_language(client, input_text, preferred_language, source_type="text"):
    """Summarizes the input text using Groq LLM in the preferred language."""
    if not client: return "Error: Groq API client not available for summarization."
    if not input_text or not input_text.strip() or input_text.startswith("[Error"):
        return f"No valid text provided from {source_type} for summarization."

    text_to_process = input_text[:MAX_TEXT_PROCESSING_CHARS]
    truncated = len(input_text) > MAX_TEXT_PROCESSING_CHARS

    lang_map = {
        "simplified_english": "simplified English using basic vocabulary and simple sentence structures.",
        "hindi": "Hindi language using Devanagari script.",
        "hinglish": "Hinglish (mix of Hindi and English, with Hindi words written in Roman script).",
        "punjabi": "Punjabi language using Gurmukhi script.",
    }
    system_prompt = f"You are a helpful assistant. Summarize the following text in {lang_map.get(preferred_language, 'English')}."
    user_prompt = f"Please provide a comprehensive summary (3-5 paragraphs) of this text extracted from {source_type}:\n\n{text_to_process}"
    if truncated: user_prompt += "\n\n[Note: The original text was truncated.]"

    try:
        response = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7, max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error (Summarization): {e}")
        return f"Error generating summary: {e}"

def extract_text_from_image_ocr(image_file_obj):
    """Extracts text from an image file object using Tesseract OCR."""
    try:
        import pytesseract
        image_file_obj.seek(0)
        image = Image.open(image_file_obj)
        if image.mode != 'RGB': image = image.convert('RGB')
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else "[OCR: No text detected]"
    except ImportError:
        st.error("Pytesseract library/engine not installed. Cannot perform OCR.")
        return "[Error: OCR Engine Not Found]"
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return f"[Error during OCR: {e}]"

def analyze_image_with_groq_vision(groq_client, image_file_obj, preferred_language):
    """Analyzes an image using Groq Vision API, falls back to OCR."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or not groq_client: return "[Error: Groq API Key/Client not configured for Vision]"

    try:
        image_file_obj.seek(0)
        base64_image = base64.b64encode(image_file_obj.read()).decode("utf-8")
        mime_type = Image.open(image_file_obj).format.lower()
        if mime_type not in ['jpeg', 'png', 'gif', 'webp']: mime_type = 'jpeg'

        lang_map = { "simplified_english": "simplified English...", "hindi": "Hindi...", "hinglish": "Hinglish...", "punjabi": "Punjabi..."}
        instruction = f"Describe this image, including any visible text. Respond in {lang_map.get(preferred_language, 'English')}."

        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = { "model": GROQ_VISION_MODEL, "messages": [{"role": "user","content": [{"type": "text", "text": instruction},{"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{base64_image}"}}]}],"temperature": 0.7, "max_tokens": 1500 }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=45) # Added timeout

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.warning(f"Groq Vision API error (code: {response.status_code}). Falling back to OCR.")
            extracted_text = extract_text_from_image_ocr(image_file_obj)
            if extracted_text.startswith("[Error") or extracted_text == "[OCR: No text detected]":
                return f"[Vision API Error: {response.status_code}. OCR fallback also failed: {extracted_text}]"
            # Summarize the OCR text if fallback succeeded
            return summarize_text_in_language(groq_client, extracted_text, preferred_language, "an image (OCR fallback)")

    except Exception as e:
        st.error(f"Error in Groq vision processing: {e}")
        return f"[Error: Vision processing failed: {e}]"

def analyze_image_with_gemini_vision(image_bytes, preferred_language, command=""):
    """Analyzes image bytes using Gemini Vision API."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key: return "[Error: Gemini API key not configured for Vision]"

    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = "image/jpeg" # Assume JPEG for bytes

        instruction = command if command else "Identify and describe what is shown in this image."
        lang_map = { "simplified_english": "simplified English...", "hindi": "Hindi...", "hinglish": "Hinglish...", "punjabi": "Punjabi..."}
        instruction += f" Respond in {lang_map.get(preferred_language, 'English')}."

        headers = {"Content-Type": "application/json"} # Key goes in params for v1beta
        params = {"key": gemini_key}
        payload = { "contents": [{"parts": [{"text": instruction},{"inline_data": {"mime_type": mime_type, "data": base64_image}}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500} }
        api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent"

        response = requests.post(api_endpoint, headers=headers, params=params, json=payload, timeout=45) # Added timeout

        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                 parts = result["candidates"][0].get("content", {}).get("parts", [])
                 if parts: return parts[0].get("text", "[Gemini Vision: No text in response]")
                 else: return "[Gemini Vision: No content parts in response]"
            else:
                 feedback = result.get('promptFeedback', {})
                 reason = feedback.get('blockReason', 'Unknown')
                 ratings = feedback.get('safetyRatings', [])
                 return f"[Gemini Vision Error: Response blocked/invalid. Reason: {reason}. Ratings: {ratings}]"
        else:
            return f"[Gemini Vision API Error: {response.status_code} - {response.text}]"

    except Exception as e:
        st.error(f"Error in Gemini vision processing: {e}")
        return f"[Error: Gemini Vision processing failed: {e}]"


# --- Voice & Camera Specific Functions ---

def speech_to_text(status_callback):
    """Converts speech to text, uses callback for status updates."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        status_callback("Listening...")
        try:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            status_callback("No speech detected.")
            return None
        except Exception as e:
            status_callback(f"Mic error: {e}")
            return None

    status_callback("Processing speech...")
    try:
        text = r.recognize_google(audio)
        status_callback(f"Heard: {text}")
        return text
    except sr.UnknownValueError:
        status_callback("Could not understand audio.")
    except sr.RequestError as e:
        status_callback(f"Google API unavailable: {e}. Trying offline...")
        try:
            text = r.recognize_sphinx(audio) # Requires pocketsphinx
            status_callback(f"Heard (offline): {text}")
            return text
        except sr.UnknownValueError: status_callback("Offline recognition failed.")
        except Exception as sphinx_e: status_callback(f"Offline error: {sphinx_e}")
    except Exception as e:
        status_callback(f"Speech recognition error: {e}")
    return None

def text_to_speech(text):
    """Converts text to speech using gTTS and plays it (runs in background thread)."""
    if not text or not text.strip(): return

    def play_audio():
        try:
            text_to_speak = text[:400].replace("*", "").replace("#", "")
            tts = gTTS(text=text_to_speak, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
                tts.save(temp_filename)

            if os.name == 'nt': os.startfile(temp_filename)
            else:
                player_found = False
                for player in ["mpg123 -q", "afplay", "play -q"]: # Try common players
                    cmd = f"{player} {temp_filename}"
                    if os.system(cmd) == 0:
                        player_found = True
                        break
                if not player_found:
                    print(f"WARN: No suitable audio player found for '{text_to_speak[:20]}...'. Install mpg123, afplay, or SoX (play).")

            time.sleep(5) # Wait for playback to likely start
            if os.path.exists(temp_filename): os.remove(temp_filename)
        except Exception as e:
            print(f"Error in TTS playback thread: {e}")
            

    thread = threading.Thread(target=play_audio, daemon=True)
    thread.start()

