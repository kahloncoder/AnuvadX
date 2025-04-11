# AnuvadX - Multilingual AI Assistant

## Overview

AnuvadX is an innovative multilingual AI assistant designed to bridge communication gaps, particularly within the diverse linguistic landscape of India. Recognizing that users interact in various languages like Hindi, Punjabi, Simple English, and Hinglish, AnuvadX provides seamless interaction, translation, and content summarization across text, documents, images, and real-time voice/camera inputs. It leverages powerful Large Language Models (LLMs) via APIs like Groq and Google Gemini to offer a unified and efficient solution.

## Why AnuvadX?

India's rich linguistic diversity presents unique challenges for information access. AnuvadX addresses this by offering:

*   **Inclusion and Accessibility:** Supporting multiple Indian languages and interaction modes ensures vital information and digital tools are available to a broader audience.
*   **Efficiency:** Utilizing robust LLM APIs (Groq Llama 3, Google Gemini) allows for complex tasks like translation, summarization, and vision analysis to be handled quickly and effectively within a single application.
*   **Multi-Modal Interaction:** Integrates text chat, PDF processing, image analysis, and real-time voice and camera capabilities, making information processing versatile and intuitive.
*   **Integrated Solution:** Combines multiple AI functionalities into one cohesive Streamlit application, showcasing how different services can work together.

## Key Features and Their Benefits

AnuvadX offers a suite of features accessible through intuitive tabs:

**1. 💬 Intelligent Chatbot**
*   **What It Does:** Detects the input language and responds conversationally in the user's preferred language (Simplified English, Hindi, Hinglish, or Punjabi) using Groq's Llama 3 model.
*   **How It Helps:** Facilitates natural communication for users regardless of their primary language, useful in education, customer support, and daily interactions.

**2. 📄 PDF Summarizer**
*   **What It Does:** Extracts text from uploaded PDF documents (using PyPDF2) and generates a concise summary in the user's chosen language (using Groq).
*   **How It Helps:** Saves time by quickly providing the essence of lengthy reports, research papers, or articles.

**3. 🖼 Image Analysis**
*   **What It Does:** Analyzes uploaded images using vision models (Gemini Vision or Groq Llama 3 Vision) to describe content or extract text, providing results in the preferred language. Falls back to OCR (Tesseract) if vision models fail or are unavailable.
*   **How It Helps:** Allows users to understand images containing text (signs, notes, diagrams) or get descriptions of visual content without needing separate tools.

**4. 🔤 Word-for-Word Translation**
*   **What It Does:** Takes text input and provides a detailed word-for-word translation along with a summary meaning in the selected language (using Groq).
*   **How It Helps:** Useful for language learners or situations requiring precise understanding of specific terms alongside the overall context.

**5. 📜 English-to-Punjabi PDF Translator (New Feature)**
*   **What It Does:** Translates entire **English** PDF documents, including text and tables (using Camelot), into **Punjabi (Gurmukhi script)**. Generates downloadable translated PDF (preserving layout via ReportLab) and TXT files. Requires **Google Gemini API Key**.
*   **How It Helps:** Provides high-fidelity document translation, preserving structure like tables, crucial for official documents, educational materials, and reports needing accurate Punjabi versions.

**6. 🎤 Voice & Camera Assistant**
*   **What It Does:** Enables hands-free interaction via voice commands (using SpeechRecognition & gTTS) and analyzes real-time video feed from the camera (using OpenCV and Gemini/Groq Vision). Users can ask questions, request analysis of objects/text shown to the camera, etc.
*   **How It Helps:** Offers an interactive, multi-modal experience for on-the-go analysis, accessibility for visually impaired users, and a dynamic way to query the real world.

AnuvadX/
├── app.py # Main Streamlit application file, includes UI and logic for most tabs
├── pdf_translator_en_pa.py # Module containing the En->Pa PDF translation logic
├── requirements.txt # Python dependencies
├── .env # File for storing API keys (GROQ_API_KEY, GEMINI_API_KEY)
├── NotoSansGurmukhi-Regular.ttf # Required font for Punjabi PDF output
├── NotoSansGurmukhi-Bold.ttf # Required font for Punjabi PDF output
└── README.md # This file

## Technical Stack

*   **Core Framework:** Streamlit
*   **Programming Language:** Python
*   **LLM & Vision APIs:**
    *   Groq API (Llama 3 Models for Chat, Summarization, Word Translation, Vision Fallback)
    *   Google Gemini API (Gemini Flash/Pro for En->Pa Translation, Gemini Pro Vision for Image/Camera Analysis)
*   **PDF Processing:**
    *   PyPDF2 (Basic Text Extraction)
    *   Camelot-py (Table Extraction for En->Pa Translator)
    *   ReportLab (Generating Translated PDF Output)
*   **Image Processing:**
    *   OpenCV (Camera Feed Handling)
    *   Pillow (Image Handling)
    *   Pytesseract (OCR Fallback for Image Analysis)
*   **Voice Processing:**
    *   SpeechRecognition (Input, supports Google Cloud Speech, Sphinx offline)
    *   gTTS (Google Text-to-Speech for Audio Output)
*   **Other Key Libraries:** Pandas, Numpy, Langdetect, python-dotenv, Requests

## Project Structure

The project follows a Streamlit application structure:

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/AnuvadX.git # Replace with your actual repo URL
    cd AnuvadX
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have Python 3.8+ installed)*

3.  **Install System Dependencies:**
    *   **Tesseract OCR Engine:** Required for the image analysis OCR fallback (`pytesseract`).
        *   Linux: `sudo apt-get update && sudo apt-get install tesseract-ocr`
        *   Mac: `brew install tesseract`
        *   Windows: Download installer from the official Tesseract GitHub repository.
    *   **Ghostscript:** Required for table extraction (`camelot-py`).
        *   Linux: `sudo apt-get install ghostscript`
        *   Mac: `brew install ghostscript`
        *   Windows: Download installer from the Ghostscript website.
    *   **PortAudio:** Required for microphone access (`SpeechRecognition`).
        *   Linux: `sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev`
        *   Mac: `brew install portaudio`
    *   **mpg123 (Optional):** Recommended for audio playback (`gTTS`) on Linux/Mac.
        *   Linux: `sudo apt-get install mpg123`
        *   Mac: `brew install mpg123`

4.  **Download Punjabi Fonts:**
    *   Download `NotoSansGurmukhi-Regular.ttf` and `NotoSansGurmukhi-Bold.ttf`.
    *   Place these `.ttf` files directly in the project's root directory (where `app.py` is located). You can find them on Google Fonts or related repositories.

5.  **Configure API Keys:**
    *   Create a file named `.env` in the project root directory.
    *   Add your API keys to the `.env` file:
        ```ini
        GROQ_API_KEY=your_groq_api_key_here
        GEMINI_API_KEY=your_google_gemini_api_key_here
        ```
    *   *Note: The Gemini API key is crucial for the En->Pa PDF Translator and enhances vision capabilities.* You can also enter keys via the Streamlit sidebar if the `.env` file is not used.

6.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    *   The application will open in your web browser.

## Usage Guidelines

1.  **API Keys:** Ensure API keys are set in `.env` or entered in the sidebar. Groq is needed for chat/summaries; Gemini is needed for En->Pa PDF translation and preferred for vision.
2.  **Language Selection:** Choose your desired response language in the sidebar.
3.  **Feature Tabs:**
    *   **Chat:** Type or speak for conversation.
    *   **PDF Summary:** Upload PDF for text extraction & summary (uses Groq).
    *   **Image Analysis:** Upload image for description/text extraction (uses Gemini/Groq Vision).
    *   **Word Translation:** Enter text for word-for-word translation (uses Groq).
    *   **En->Pa PDF Translator:** Upload **English PDF** for full translation to **Punjabi** (uses Gemini). Download translated PDF/TXT.
    *   **Voice & Camera:** Interact using voice and analyze camera feed.

## Contributing

We welcome contributions! Please follow standard GitHub practices:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Commit your changes.
4.  Push to your branch.
5.  Submit a pull request.

Please report any bugs or suggest enhancements using GitHub Issues.

## Final Thoughts

AnuvadX aims to be more than just a tool; it's a step towards digital inclusivity in India. By leveraging advanced AI, we hope to empower users, break down language barriers, and make information accessible to everyone.

For more information or inquiries, please open an issue on GitHub.

**AnuvadX – Connecting India, One Language at a Time!**
