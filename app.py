import streamlit as st
import os
from dotenv import load_dotenv
import io
from PIL import Image
import threading
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- Import Utility Functions & PDF Translator Module ---
import utils 
try:
    import pdf_translator_en_pa
except ImportError:
    st.error("ERROR: PDF Translator module 'pdf_translator_en_pa.py' not found.")
    pdf_translator_en_pa = None

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize API Clients (using cached functions from utils) ---
groq_client = utils.get_groq_client()

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Multilingual AI Assistant", page_icon="🌍", layout="wide")

# --- Initialize Session State ---

if "messages" not in st.session_state: st.session_state.messages = []
if "listening" not in st.session_state: st.session_state.listening = False
if "camera_active" not in st.session_state: st.session_state.camera_active = False
if "voice_output_text" not in st.session_state: st.session_state.voice_output_text = "Press 'Start Listening' or use tabs."
if "preferred_language" not in st.session_state: st.session_state.preferred_language = "simplified_english"
if "video_processor" not in st.session_state: st.session_state.video_processor = None # Initialize processor state

# --- UI Helper Functions (Specific to this app.py) ---
def _add_message_to_chat(role, content):
    """Adds a message to the session state chat history."""
    st.session_state.messages.append({"role": role, "content": content})

def _display_chat_messages():
    """Displays the chat messages stored in session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def update_voice_status(status_text):
    """Callback function to update voice status in session state and rerun."""
    st.session_state.voice_output_text = status_text
    try:
        st.experimental_rerun()
    except Exception as e:
        print(f"Note: Rerun failed during voice status update: {e}")
        pass


# --- Voice & Camera Handling (Remains in app.py due to UI coupling) ---

class VideoProcessor(VideoTransformerBase):
    """Processes video frames and stores the latest."""
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.frame_lock:
            self.latest_frame = img.copy()

        height, width = img.shape[:2]
        center_area = int(min(width, height) * 0.7); x1, y1 = (width - center_area) // 2, (height - center_area) // 2
        x2, y2 = x1 + center_area, y1 + center_area; cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Aim here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame.from_ndarray(img, format="bgr24")

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame

def _handle_voice_command(text, preferred_language, video_processor):
    """Orchestrates processing of recognized voice command using utils functions."""
    is_camera_command = any(kw in text.lower() for kw in ["analyze", "capture", "scan", "what is this", "camera", "show"])

    if is_camera_command:
        latest_frame = video_processor.get_latest_frame() if video_processor else None
        if latest_frame is not None:
            update_voice_status("Analyzing camera view...")
            success, encoded_img = cv2.imencode(".jpg", latest_frame)
            if success:
                img_bytes = encoded_img.tobytes()
                command = text 
                result = "[Analysis Pending...]"
                if os.getenv("GEMINI_API_KEY"):
                    result = utils.analyze_image_with_gemini_vision(img_bytes, preferred_language, command)
                elif os.getenv("GROQ_API_KEY"):
                    update_voice_status("Using Groq Vision...")
                    img_bytes_io = io.BytesIO(img_bytes); img_bytes_io.name = "voice_capture.jpg"
                    result = utils.analyze_image_with_groq_vision(groq_client, img_bytes_io, preferred_language)
                else:
                    result = "[Error: No Vision API key configured]"

                _add_message_to_chat("user", f"[Voice Command & Camera] {text}")
                _add_message_to_chat("assistant", result)
                update_voice_status("Analysis complete.")
                utils.text_to_speech(result) 
            else:
                 update_voice_status("Failed to process camera frame.")
                 utils.text_to_speech("Failed to process camera frame.")
        else:
            update_voice_status("Camera not ready or no frame captured.")
            utils.text_to_speech("Camera not ready.")
    else:
        # Regular chat command
        if groq_client:
            response = utils.get_response_in_language(groq_client, text, preferred_language, st.session_state.messages)
            _add_message_to_chat("user", f"[Voice] {text}")
            _add_message_to_chat("assistant", response)
            update_voice_status("Responded.")
            utils.text_to_speech(response)
        else:
            update_voice_status("Chat API not configured.")
            utils.text_to_speech("The chat API is not configured.")


def listen_and_process_thread(preferred_language, video_processor):
    """Target function for the listening thread."""
    try:
        recognized_text = utils.speech_to_text(status_callback=update_voice_status)
        if recognized_text:
            _handle_voice_command(recognized_text, preferred_language, video_processor)
    except Exception as e:
        print(f"ERROR in listen_and_process_thread: {e}")
        update_voice_status(f"Error in voice processing: {e}") 
    finally:
        st.session_state.listening = False
        try:
            st.experimental_rerun()
        except Exception: pass # Ignore if rerun fails

def start_listening_thread(preferred_language, video_processor):
    """Starts the voice listening thread."""
    if not st.session_state.listening:
        st.session_state.listening = True
        thread = threading.Thread(target=listen_and_process_thread,
                                  args=(preferred_language, video_processor),
                                  daemon=True)
        thread.start()
        st.experimental_rerun() 


# --- Streamlit UI ---

st.title("🌍 Multilingual AI Assistant")
st.markdown("Chat, summarize PDFs, analyze images, translate text, and use voice/camera interaction!")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("🌐 Language")
    lang_options = { "simplified_english": "Simplified English", "hindi": "Hindi (हिन्दी)", "hinglish": "Hinglish", "punjabi": "Punjabi (ਪੰਜਾਬੀ)" }
    st.session_state.preferred_language = st.radio( "Preferred response language:", options=lang_options.keys(), format_func=lang_options.get, key="pref_lang_radio_sidebar" )

    # API Configuration
    st.subheader("🔑 API Keys")
    groq_api_key_env = os.getenv("GROQ_API_KEY")
    gemini_api_key_env = os.getenv("GEMINI_API_KEY")
    # Groq Input
    if not groq_api_key_env:
        groq_input = st.text_input("Enter Groq API Key:", type="password", key="groq_key_input_sb_main")
        if groq_input: os.environ["GROQ_API_KEY"] = groq_input; st.success("Groq Key Set! Refreshing..."); time.sleep(1); st.rerun()
    else: st.success("Groq Key Loaded.", icon="✔️")
    # Gemini Input
    gemini_input = st.text_input("Gemini API Key (Needed for En->Pa PDF & Vision):", type="password", value=gemini_api_key_env or "", key="gemini_key_input_sb_main")
    if gemini_input and gemini_input != gemini_api_key_env:
        os.environ["GEMINI_API_KEY"] = gemini_input; st.success("Gemini Key Set! Refreshing..."); time.sleep(1); st.rerun()
    elif gemini_api_key_env: st.success("Gemini Key Loaded.", icon="✔️")
    else: st.warning("Gemini Key Recommended.", icon="⚠️")

    # Chat Management
    st.subheader("🗑️ Chat")
    if st.button("Clear Chat History"):
        st.session_state.messages = []; st.session_state.voice_output_text = "Chat cleared."; st.rerun()

    # Help/Info Sections
    with st.expander("🔧 Installation & Setup"): st.markdown("See `README.md` for details.") # Point to README
    with st.expander("💡 How to Use"): st.markdown("Select language, configure keys, and use the tabs!") # Simple instructions

    st.markdown("---")
    st.caption("Powered by Groq, Gemini, and Streamlit")

# --- Main Application Tabs ---
tab_names = ["💬 Chat", "📄 PDF Summary", "🖼 Image Analysis", "🔤 Word Translation", "📜 En->Pa PDF Translator", "🎤 Voice & Camera"]
tabs = st.tabs(tab_names)

# --- Tab 1: Chat ---
with tabs[0]:
    st.header("💬 Conversational Chat")
    st.markdown(f"Ask questions or chat. Responses will be in **{lang_options[st.session_state.preferred_language]}**.")
    _display_chat_messages()
    if user_input := st.chat_input("Type your message here...", key="chat_input_tab1"):
        _add_message_to_chat("user", user_input)
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = utils.get_response_in_language(groq_client, user_input, st.session_state.preferred_language, st.session_state.messages)
                st.markdown(response)
                _add_message_to_chat("assistant", response)
        detected_lang = utils.detect_language(user_input) # Use util function
        st.sidebar.info(f"Last input lang: {detected_lang}")

# --- Tab 2: PDF Summary ---
with tabs[1]:
    st.header("📄 PDF Summarizer")
    st.markdown(f"Upload PDF for text extraction & summary in **{lang_options[st.session_state.preferred_language]}**.")
    uploaded_pdf_summary = st.file_uploader("Upload PDF for Summary", type="pdf", key="pdf_uploader_summary_tab2")
    if uploaded_pdf_summary:
        if st.button("Extract and Summarize PDF", key="summarize_pdf_button_tab2"):
            with st.spinner("Processing PDF..."):
                pdf_bytes = uploaded_pdf_summary.getvalue()
                pdf_text = utils.extract_text_from_pdf_basic(pdf_bytes) # Use util
                if pdf_text.startswith("[Error"): st.error(pdf_text)
                elif not pdf_text or pdf_text == "[No text extracted from PDF]": st.warning(pdf_text)
                else:
                    st.success("Text extracted.")
                    with st.expander("View Extracted Text & Summary", expanded=True):
                        st.text_area("Extracted Text (Preview)", pdf_text[:3000] + "..." if len(pdf_text)>3000 else pdf_text, height=200)
                        st.markdown("---")
                        st.markdown("### Summary")
                        with st.spinner("Generating summary..."):
                            summary = utils.summarize_text_in_language(groq_client, pdf_text, st.session_state.preferred_language, f"'{uploaded_pdf_summary.name}'") # Use util
                            st.markdown(summary)

# --- Tab 3: Image Analysis ---
with tabs[2]:
    st.header("🖼 Image Analysis")
    st.markdown(f"Upload an image for analysis in **{lang_options[st.session_state.preferred_language]}**.")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="image_uploader_tab3")
    if uploaded_image:
        img_disp_col, img_an_col = st.columns([2, 3])
        with img_disp_col:
             try: st.image(Image.open(uploaded_image), caption="Uploaded", use_column_width=True)
             except Exception as e: st.error(f"Cannot display image: {e}")
        with img_an_col:
            if st.button("Analyze Image Content", key="analyze_image_button_tab3"):
                with st.spinner("Analyzing image..."):
                    analysis_result = "[Analysis Pending...]"
                    # Prioritize Gemini if key exists
                    if os.getenv("GEMINI_API_KEY"):
                        analysis_result = utils.analyze_image_with_gemini_vision(uploaded_image.getvalue(), st.session_state.preferred_language)
                    elif os.getenv("GROQ_API_KEY") and groq_client:
                        analysis_result = utils.analyze_image_with_groq_vision(groq_client, uploaded_image, st.session_state.preferred_language)
                    else: analysis_result = "[Error: No Vision API (Gemini/Groq) configured]"

                    st.markdown("#### Analysis Result:")
                    if analysis_result.startswith("[Error"): st.error(analysis_result)
                    else: st.success(analysis_result)

# --- Tab 4: Word Translation ---
with tabs[3]:
    st.header("🔤 Word-for-Word Translation")
    st.markdown(f"Translate text word-by-word into **{lang_options[st.session_state.preferred_language]}**.")
    input_text_translate = st.text_area("Enter text to translate:", height=150, key="text_translate_input_tab4")
    if st.button("Translate Word-for-Word", key="translate_word_button_tab4") and input_text_translate:
        with st.spinner("Translating..."):
            translation_result = utils.get_word_for_word_translation(groq_client, input_text_translate, st.session_state.preferred_language) # Use util
            st.markdown("#### Translation Result:")
            st.markdown(translation_result)

# --- Tab 5: En->Pa PDF Translator ---
with tabs[4]:
    st.header("📜 English to Punjabi PDF Translator")
    st.markdown("Upload **English PDF** → Translate to **Punjabi**.")
    st.warning("**Requires Google Gemini API Key.**", icon="🔑")

    if pdf_translator_en_pa is None: st.error("Translator module not loaded.")
    elif not os.getenv("GEMINI_API_KEY"): st.error("Configure Gemini API Key in sidebar.")
    else:
        gemini_model_pdf = pdf_translator_en_pa.setup_gemini_api_pdf() # Setup uses cache
        punjabi_fonts_pdf = pdf_translator_en_pa.setup_punjabi_font_pdf() # Setup uses cache

        if gemini_model_pdf and punjabi_fonts_pdf:
            uploaded_pdf_en_pa = st.file_uploader("Upload English PDF for Punjabi Translation", type="pdf", key="pdf_uploader_en_pa_tab5")
            if uploaded_pdf_en_pa:
                with st.expander("Options", expanded=False):
                     save_intermediate = st.checkbox("Save intermediate files (debug)", key="save_intermediate_pa_tab5")
                     max_retries = st.slider("Max API Retries", 0, 10, 5, key="max_retries_pa_tab5")
                if st.button("🚀 Start En->Pa Translation", key="start_en_pa_translation_tab5"):
                    pdf_bytes = uploaded_pdf_en_pa.getvalue()
                    fname = uploaded_pdf_en_pa.name
                    base_fname = os.path.splitext(fname)[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Call the module's main function
                    pdf_out_bytes, txt_out_bytes = pdf_translator_en_pa.run_pdf_translation_process(
                        pdf_bytes, gemini_model_pdf, punjabi_fonts_pdf, save_intermediate, max_retries )
                    if pdf_out_bytes and txt_out_bytes:
                        st.markdown("---"); st.subheader("✅ Download Translated Files:")
                        dcol1, dcol2 = st.columns(2)
                        dcol1.download_button("PDF (Punjabi)", pdf_out_bytes, f"{base_fname}_punjabi_{timestamp}.pdf", "application/pdf")
                        dcol2.download_button("TXT (Punjabi)", txt_out_bytes, f"{base_fname}_punjabi_{timestamp}.txt", "text/plain")
                    else: st.error("Translation process failed.")
        else: st.error("Translator components failed to initialize (API/Fonts).")

# --- Tab 6: Voice & Camera ---
with tabs[5]:
    st.header("🎤 Voice & Camera Assistant")
    st.markdown("Use voice commands and camera for real-time interaction.")

    # Voice Section
    st.subheader("🗣️ Voice")
    vcol1, vcol2 = st.columns([3, 1])
    vcol1.info(st.session_state.voice_output_text) # Display status
    with vcol2:
        # Get video processor instance (initialized below if needed)
        video_processor_instance = st.session_state.get("video_processor")
        if st.session_state.listening: st.button("🔴 Listening...", disabled=True)
        else:
             if st.button("🎤 Start Listening", key="start_listening_tab6"):
                 # Pass the potentially existing processor instance
                 start_listening_thread(st.session_state.preferred_language, video_processor_instance)

    st.markdown("---")
    # Camera Section
    st.subheader("📸 Camera")
    ccol1, ccol2 = st.columns([2, 1])
    with ccol1:
        st.markdown("Enable camera:")
        # Initialize processor if it doesn't exist
        if st.session_state.video_processor is None:
             st.session_state.video_processor = VideoProcessor()

        webrtc_ctx = webrtc_streamer( key="camera_streamer_tab6", rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                                     video_processor_factory=lambda: st.session_state.video_processor, # Return existing instance
                                     media_stream_constraints={"video": True, "audio": False}, async_processing=True )
        st.session_state.camera_active = webrtc_ctx.state.playing
        if not st.session_state.camera_active: st.caption("Camera off.")

    with ccol2:
        st.markdown("**Manual Capture:**")
        # Ensure processor exists *and* camera is active before enabling button
        if st.session_state.camera_active and st.session_state.video_processor:
            if st.button("👁️ Analyze View", key="manual_capture_tab6"):
                latest_frame = st.session_state.video_processor.get_latest_frame()
                if latest_frame is not None:
                    with st.spinner("Analyzing view..."):
                        success, encoded_img = cv2.imencode(".jpg", latest_frame)
                        if success:
                            result = "[Analysis Pending...]"
                            img_bytes = encoded_img.tobytes()
                            # Prioritize Gemini
                            if os.getenv("GEMINI_API_KEY"):
                                result = utils.analyze_image_with_gemini_vision(img_bytes, st.session_state.preferred_language, "Analyze what's shown")
                            elif groq_client:
                                img_bytes_io = io.BytesIO(img_bytes); img_bytes_io.name="capture.jpg"
                                result = utils.analyze_image_with_groq_vision(groq_client, img_bytes_io, st.session_state.preferred_language)
                            else: result = "[Error: No Vision API configured]"
                            _add_message_to_chat("user", "[Camera] Manual analysis requested.")
                            _add_message_to_chat("assistant", result)
                            st.success("Analysis added to chat!")
                            st.rerun() # Show updated chat
                        else: st.error("Failed encoding frame.")
                else: st.warning("No frame available yet.")
        else: st.caption("Start camera first.")