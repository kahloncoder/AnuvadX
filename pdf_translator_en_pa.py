# pdf_translator_en_pa.py

import streamlit as st
import os
import io
import re
import traceback
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import camelot
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import google.generativeai as genai
import json
from datetime import datetime
from time import sleep
import random
import tempfile

# --- Constants ---
FONT_REGULAR_PATH = "NotoSansGurmukhi-Regular.ttf"
FONT_BOLD_PATH = "NotoSansGurmukhi-Bold.ttf"
FONT_REGULAR_NAME = "NotoSansGurmukhi"
FONT_BOLD_NAME = "NotoSansGurmukhiBold"
OUTPUT_DIR_PA = "translated_output_en_pa" # Specific output dir
CACHE_DIR_PA = "translation_cache_en_pa"
PROGRESS_DIR_PA = "translation_progress_en_pa"

# --- Gemini API Setup for this Feature ---
# Use cache_resource to avoid re-initializing the model on every interaction within the tab
@st.cache_resource
def setup_gemini_api_pdf():
    """Set up the Gemini API specifically for PDF translation."""
    try:
        # Use the Gemini key stored in environment variables by the main app
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ Gemini API Key not found. Please configure it in the sidebar for this feature.")
            return None

        genai.configure(api_key=api_key)
        # Use a model known for long context if available, like 1.5 Pro, but Flash is okay too.
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-1.5-pro-latest'
        st.success("✅ Gemini API for PDF Translation Initialized.")
        return model
    except Exception as e:
        st.error(f"❌ Gemini API (PDF Translator) Connection Error: {e}")
        return None

# --- Font Setup ---
@st.cache_resource
def setup_punjabi_font_pdf():
    """Load and register Punjabi font for PDF generation."""
    fonts_registered = {"regular": None, "bold": None}
    try:
        # Check for Regular Font
        if not os.path.exists(FONT_REGULAR_PATH):
            st.error(f"❌ Critical Error: Punjabi font file '{FONT_REGULAR_PATH}' not found.")
            st.error("Please download 'NotoSansGurmukhi-Regular.ttf' and place it in the application directory.")
            return None # Cannot proceed without regular font
        pdfmetrics.registerFont(TTFont(FONT_REGULAR_NAME, FONT_REGULAR_PATH))
        fonts_registered["regular"] = {"path": FONT_REGULAR_PATH, "name": FONT_REGULAR_NAME}
        st.info(f"ℹ️ Registered Punjabi regular font: {FONT_REGULAR_NAME}")

        # Try to register Bold Font
        if os.path.exists(FONT_BOLD_PATH):
            try:
                pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, FONT_BOLD_PATH))
                fonts_registered["bold"] = {"path": FONT_BOLD_PATH, "name": FONT_BOLD_NAME}
                st.info(f"ℹ️ Registered Punjabi bold font: {FONT_BOLD_NAME}")
            except Exception as bold_e:
                st.warning(f"⚠️ Could not register bold font from {FONT_BOLD_PATH}: {bold_e}. Using regular font for bold text.")
                fonts_registered["bold"] = fonts_registered["regular"] # Fallback
        else:
            st.warning(f"⚠️ Bold font file '{FONT_BOLD_PATH}' not found. Using regular font for bold text.")
            fonts_registered["bold"] = fonts_registered["regular"] # Fallback

        return fonts_registered

    except Exception as e:
        st.error(f"❌ Error setting up Punjabi fonts: {e}")
        return None

# --- Core Logic Functions (Adapted for Streamlit UI feedback) ---

def api_call_with_backoff_pdf(model, prompt, max_retries=5, initial_wait=2, step_name="API Call"):
    """Make API calls with exponential backoff (specific for this module)."""
    retries = 0
    wait_time = initial_wait
    while retries <= max_retries:
        try:
            # Use st.spinner for feedback during API calls
            with st.spinner(f"🔄 {step_name} - Attempt {retries+1}/{max_retries+1}..."):
                response = model.generate_content(prompt)
                # Add small delay to prevent immediate hammering if many short calls
                sleep(0.5 + random.random() * 0.5)
            # Check for safety ratings or blocked content if applicable
            if not response.candidates:
                 raise Exception("API response blocked or invalid.")
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            # Refined check for rate limit/quota errors
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str or "resource has been exhausted" in error_str:
                retries += 1
                if retries > max_retries:
                    st.error(f"❌ Max retries ({max_retries}) reached for {step_name}. API call failed.")
                    raise e
                actual_wait = wait_time + (random.random() * wait_time * 0.2)
                st.warning(f"⏳ Rate limit likely hit for {step_name}. Waiting {actual_wait:.1f}s before retry {retries}/{max_retries}...")
                sleep(actual_wait)
                wait_time *= 2 # Exponential backoff
            else: # Handle other errors
                st.error(f"❌ Error during {step_name} (Retry {retries+1}): {e}")
                raise e
    raise Exception(f"Maximum retries exceeded for {step_name}")


def generate_summary_pdf(text, model, language="English", max_retries=5):
    """Generate summary (specific for this module)."""
    step_name = f"Generating {language} Summary"
    try:
        # Truncate input text to avoid exceeding model limits significantly
        truncated_text = text[:15000] # Increased limit slightly for Gemini Pro potentially
        if len(text) > 15000:
            st.sidebar.warning("Input text for summary was truncated.")

        if language == "English":
            prompt = f"""Create a brief summary (2-3 short paragraphs) of the key points and main ideas in the following text. Focus on conciseness and clarity. TEXT TO SUMMARIZE:\n\n{truncated_text}\n\nOUTPUT ONLY THE SUMMARY."""
        else:  # Punjabi
            prompt = f"""Write a brief summary (2-3 short paragraphs) of the key points and main ideas in the following Punjabi text, using Gurmukhi script. Focus on conciseness and clarity. TEXT TO SUMMARIZE:\n\n{truncated_text}\n\nOUTPUT ONLY THE SUMMARY IN PUNJABI."""
        return api_call_with_backoff_pdf(model, prompt, max_retries=max_retries, step_name=step_name)
    except Exception as e:
        st.error(f"❌ Error {step_name}: {e}")
        return f"[{language} Summary generation failed]"

def summarize_table_pdf(df, model, language="English", max_retries=5):
    """Generate table summary (specific for this module)."""
    step_name = f"Summarizing Table ({language})"
    try:
        # Attempt to convert DataFrame to string, handle potential large tables
        try:
            table_str = df.to_string(max_rows=50, max_cols=10) # Limit string representation
            if len(df) > 50 or len(df.columns) > 10:
                 table_str += "\n\n[Note: Table truncated for summary generation]"
        except Exception as str_err:
             st.warning(f"Could not convert table fully to string for summary: {str_err}")
             # Provide basic info if conversion fails
             table_str = f"Table with {len(df)} rows and {len(df.columns)} columns. Headers: {', '.join(map(str, df.columns[:5]))}..."

        if language == "English":
            prompt = f"""Analyze this table data and provide a brief 2-3 sentence summary of what it likely represents. TABLE DATA:\n\n{table_str}\n\nOUTPUT ONLY THE TABLE SUMMARY."""
        else:  # Punjabi
            prompt = f"""Analyze this table data and provide a brief 2-3 sentence summary in Punjabi (Gurmukhi script) of what it likely represents. TABLE DATA:\n\n{table_str}\n\nOUTPUT ONLY THE TABLE SUMMARY IN PUNJABI."""
        return api_call_with_backoff_pdf(model, prompt, max_retries=max_retries, step_name=step_name)
    except Exception as e:
        st.error(f"❌ Error {step_name}: {e}")
        return f"[{language} Table Summary generation failed]"


def extract_tables_and_text_pdf(pdf_content):
    """Extract content (specific for this module)."""
    results = {'tables': [], 'page_texts': [], 'metadata': {}}
    temp_pdf_path = None

    try:
        # PyPDF2 for text and metadata
        reader = PdfReader(io.BytesIO(pdf_content))
        num_pages = len(reader.pages)
        st.write(f"📄 PDF has {num_pages} pages.")
        results['metadata'] = {k: v for k, v in reader.metadata.items()} if reader.metadata else {}

        text_extract_bar = st.progress(0, text="Extracting Text (PyPDF2)...")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or "" # Ensure it's a string
                results['page_texts'].append((i + 1, page_text))
            except Exception as text_err:
                st.sidebar.error(f"Page {i+1}: Error extracting text: {text_err}")
                results['page_texts'].append((i + 1, f"[Text Extraction Error: {text_err}]"))
            text_extract_bar.progress((i + 1) / num_pages, text=f"Extracting Text... Page {i+1}/{num_pages}")
        text_extract_bar.empty()

        # Camelot for tables
        st.write("🔍 Extracting tables with Camelot...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_content)
            temp_pdf_path = tmpfile.name

        extracted_tables = {}
        # Combine lattice and stream attempts more robustly
        for flavor in ['lattice', 'stream']:
            with st.spinner(f"Running Camelot ({flavor})..."):
                try:
                    tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor=flavor, suppress_stdout=True, line_scale=40 if flavor=='lattice' else 15)
                    st.write(f"  - Found {tables.n} potential tables ({flavor}).")
                    for table in tables:
                        if table.df.size > 0 and table.parsing_report['accuracy'] > 80: # Basic quality filter
                            page_num = table.page
                            # Avoid adding duplicate or very similar tables from different methods
                            is_new = True
                            if page_num in extracted_tables:
                                for existing_df in extracted_tables[page_num]:
                                     # Simple check: if shapes are identical, assume duplicate
                                     if existing_df.shape == table.df.shape:
                                         is_new = False
                                         break
                            if is_new:
                                 if page_num not in extracted_tables: extracted_tables[page_num] = []
                                 extracted_tables[page_num].append(table.df)
                except Exception as e:
                    st.warning(f"⚠️ Camelot ({flavor}) may have failed or found no tables: {e}")

        # Convert to results format
        for page_num, page_tables in extracted_tables.items():
            for df in page_tables:
                # Clean table headers (remove excessive whitespace/newlines)
                df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
                results['tables'].append((page_num, df))

        st.write(f"✅ Table extraction complete: Found {len(results['tables'])} reasonably accurate tables.")

    except Exception as e:
        st.error(f"❌ Error during extraction: {e}")
        st.exception(e)
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try: os.remove(temp_pdf_path)
            except: pass
    return results

def is_complex_text_pdf(text):
    """Determine if text needs summarization (specific heuristic)."""
    if len(text) > 2500: return True
    if len(re.findall(r'\d{3,}', text)) > 10: return True # Many 3+ digit numbers
    if text.count('\n\n') > 20: return True # Many paragraphs
    return False

def translate_content_pdf(page_texts, tables, model, save_intermediate=False, max_retries=5):
    """Translate content (specific for this module)."""
    results = {'translated_texts': [], 'translated_tables': [], 'text_summaries': {}, 'table_summaries': {}}
    if save_intermediate: os.makedirs(CACHE_DIR_PA, exist_ok=True)

    total_items = len(page_texts) + len(tables)
    overall_progress = st.progress(0, text="Starting Translation...")
    items_processed = 0

    # 1. Translate page texts
    st.subheader("1. Translating Page Texts")
    texts_placeholder = st.empty() # Placeholder for page-by-page status

    for i, (page_num, text) in enumerate(page_texts):
        items_processed += 1
        progress_text = f"Processing Page {page_num}/{len(page_texts)}"
        overall_progress.progress(items_processed / total_items, text=progress_text)
        texts_placeholder.write(f"**Page {page_num}:** Translating text...")

        if not text or not text.strip():
            results['translated_texts'].append((page_num, ""))
            texts_placeholder.write(f"**Page {page_num}:** Skipped (empty).")
            continue

        try:
            needs_summary = is_complex_text_pdf(text)
            if needs_summary:
                 texts_placeholder.write(f"**Page {page_num}:** Generating English summary...")
                 english_summary = generate_summary_pdf(text, model, "English", max_retries)
                 if save_intermediate:
                     with open(f"{CACHE_DIR_PA}/page_{page_num}_eng_summary.txt", "w", encoding="utf-8") as f: f.write(english_summary)

            # Chunking (same logic)
            chunks = []
            current_chunk = ""
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                # Increased max chunk size slightly for potentially more capable models
                max_chunk_len = 9000
                if len(paragraph) > max_chunk_len:
                    if current_chunk: chunks.append(current_chunk); current_chunk = ""
                    chunks.extend([paragraph[j:j+max_chunk_len] for j in range(0, len(paragraph), max_chunk_len)])
                elif len(current_chunk) + len(paragraph) > max_chunk_len:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    if current_chunk: current_chunk += "\n\n"
                    current_chunk += paragraph
            if current_chunk: chunks.append(current_chunk)
            if not chunks and text: chunks = [text]

            translated_chunks = []
            texts_placeholder.write(f"**Page {page_num}:** Translating {len(chunks)} text chunk(s)...")
            for chunk_idx, chunk in enumerate(chunks):
                step_name = f"Page {page_num} Chunk {chunk_idx+1}/{len(chunks)}"
                prompt = f"""Translate the following English text TO PUNJABI (Gurmukhi script).

ENSURE YOU FOLLOW THESE RULES VERY STRICTLY:
1.  Translate TO Punjabi (Gurmukhi Script).
2.  PRESERVE ALL ORIGINAL FORMATTING: Keep paragraphs, line breaks, indentation, lists, bullet points (like *, -, •) exactly as they appear in the original English text.
3.  PRESERVE ALL NUMBERS, DATES, YEARS, percentages, currency symbols ($), and technical codes EXACTLY as they are. Do NOT translate numbers into Punjabi words.
4.  KEEP Proper Nouns (like names of people, organizations, specific places, technical terms) in their original English form unless a very common, standard Punjabi equivalent exists. When in doubt, keep the English word.
5.  Maintain any special characters or symbols.
6.  Output ONLY the translated Punjabi text. NO additional comments, explanations, or introductory phrases like "Here is the translation:".

ENGLISH TEXT TO TRANSLATE:
--- START ---
{chunk}
--- END ---

PUNJABI TRANSLATION (Gurmukhi Script):"""
                try:
                    translated_chunk = api_call_with_backoff_pdf(model, prompt, max_retries=max_retries, step_name=step_name)
                    translated_chunks.append(translated_chunk)
                except Exception as chunk_e:
                    st.error(f"Error translating {step_name}: {chunk_e}")
                    translated_chunks.append(f"[CHUNK TRANSLATION ERROR: {chunk_e}]")

            translated_text = '\n\n'.join(translated_chunks)

            if needs_summary:
                texts_placeholder.write(f"**Page {page_num}:** Generating Punjabi summary...")
                punjabi_summary = generate_summary_pdf(translated_text, model, "Punjabi", max_retries)
                results['text_summaries'][page_num] = punjabi_summary
                if save_intermediate:
                    with open(f"{CACHE_DIR_PA}/page_{page_num}_punjabi_summary.txt", "w", encoding="utf-8") as f: f.write(punjabi_summary)

            results['translated_texts'].append((page_num, translated_text))
            if save_intermediate:
                 with open(f"{CACHE_DIR_PA}/page_{page_num}_original.txt", "w", encoding="utf-8") as f: f.write(text)
                 with open(f"{CACHE_DIR_PA}/page_{page_num}_translated.txt", "w", encoding="utf-8") as f: f.write(translated_text)

            texts_placeholder.write(f"**Page {page_num}:** Text translation complete.")

        except Exception as page_e:
            st.error(f"Critical error processing page {page_num} text: {page_e}")
            results['translated_texts'].append((page_num, f"[PAGE TRANSLATION ERROR: {page_e}]"))

    texts_placeholder.empty() # Clear the page status area

    # 2. Translate tables
    st.subheader("2. Translating Tables")
    tables_placeholder = st.empty() # Placeholder for table status

    for i, (page_num, df) in enumerate(tables):
        items_processed += 1
        progress_text = f"Processing Table {i+1}/{len(tables)} (Page {page_num})"
        overall_progress.progress(items_processed / total_items, text=progress_text)
        tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Translating...")

        try:
            df_translated = df.copy(deep=True)

            # Summaries first
            tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Generating English summary...")
            eng_tbl_summary = summarize_table_pdf(df, model, "English", max_retries)
            if save_intermediate:
                 with open(f"{CACHE_DIR_PA}/table_page_{page_num}_{i}_eng_summary.txt", "w", encoding="utf-8") as f: f.write(eng_tbl_summary)

            # Translate Headers
            tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Translating headers...")
            new_columns = {}
            original_columns = list(df.columns) # Keep original order
            for col_idx, col in enumerate(original_columns):
                col_str = str(col).strip()
                # Skip numeric/empty/simple symbol headers
                if col_str and not re.match(r'^[\d\s.,%\-+/$£€¥#]*$', col_str):
                    step_name = f"Tbl {i+1} Header '{col_str[:20]}...'"
                    prompt = f"""Translate ONLY the following table header text from English TO Punjabi (Gurmukhi script). Keep it concise. Preserve numbers and symbols.

Header Text: "{col_str}"

Output ONLY the translated Punjabi text. No extra words."""
                    try:
                        translated_header = api_call_with_backoff_pdf(model, prompt, max_retries=max_retries, step_name=step_name)
                        new_columns[col] = translated_header
                    except Exception as header_e:
                        st.warning(f"Warn translating {step_name}: {header_e}")
                        new_columns[col] = col # Keep original on error
                else:
                    new_columns[col] = col # Keep non-translatable headers
            # Apply translation while preserving original order
            df_translated.columns = [new_columns.get(col, col) for col in original_columns]


            # Translate Cells
            tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Translating {df.shape[0]} rows...")
            total_cells = df.size
            cells_processed = 0
            # Iterate using numpy for potentially faster access? Or stick to iterrows/iloc
            for r_idx in range(df_translated.shape[0]):
                for c_idx in range(df_translated.shape[1]):
                    cell = df.iloc[r_idx, c_idx] # Get original cell for translation
                    cell_text = str(cell).strip()

                    # Skip simple cells more aggressively
                    if not cell_text or len(cell_text) < 2 or re.match(r'^[\d\s.,%\-+/$£€¥#():;<=>?@[\]^_`{|}~]*$', cell_text):
                        cells_processed += 1
                        continue

                    step_name = f"Tbl {i+1} Cell(R{r_idx},C{c_idx})"
                    prompt = f"""Translate ONLY the following table cell text from English TO Punjabi (Gurmukhi script). Keep it concise. Preserve numbers, symbols, formatting.

Cell Text: "{cell_text}"

Output ONLY the translated Punjabi text. No extra words."""
                    try:
                        translated_cell = api_call_with_backoff_pdf(model, prompt, max_retries=max_retries, step_name=step_name)
                        df_translated.iloc[r_idx, c_idx] = translated_cell
                    except Exception as cell_e:
                        st.warning(f"Warn translating {step_name}: {cell_e}")
                        # Keep original cell on error df_translated.iloc[r_idx, c_idx] = cell_text + " [ERR]"
                    cells_processed += 1
                # Update progress within table processing if needed (can be slow)
                # if r_idx % 10 == 0: tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Translating row {r_idx+1}/{df.shape[0]}...")


            tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Generating Punjabi summary...")
            punjabi_tbl_summary = summarize_table_pdf(df_translated, model, "Punjabi", max_retries)
            results['table_summaries'][f"{page_num}_{i}"] = punjabi_tbl_summary # Use unique key

            results['translated_tables'].append((page_num, df_translated, i)) # Include original index 'i'

            if save_intermediate:
                 df.to_csv(f"{CACHE_DIR_PA}/table_page_{page_num}_{i}_original.csv", index=False)
                 df_translated.to_csv(f"{CACHE_DIR_PA}/table_page_{page_num}_{i}_translated.csv", index=False)
                 with open(f"{CACHE_DIR_PA}/table_page_{page_num}_{i}_punjabi_summary.txt", "w", encoding="utf-8") as f: f.write(punjabi_tbl_summary)

            tables_placeholder.write(f"**Table {i+1} (Page {page_num}):** Translation complete.")

        except Exception as table_e:
            st.error(f"Critical error processing table {i+1} on page {page_num}: {table_e}")
            results['translated_tables'].append((page_num, pd.DataFrame({"Error": [f"Translation failed: {str(table_e)}"]}), i))

    tables_placeholder.empty() # Clear table status area
    overall_progress.empty()
    return results

# --- Output Generation Functions ---

def create_txt_output_bytes_pdf(translated_content):
    """Create TXT output bytes (specific for this module)."""
    try:
        output_io = io.StringIO()
        output_io.write("=" * 80 + "\n")
        output_io.write("ENGLISH TO PUNJABI PDF TRANSLATION\n")
        output_io.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        output_io.write("=" * 80 + "\n\n")

        # Combine text and tables by page number
        content_by_page = {}
        for page_num, text in translated_content['translated_texts']:
            if page_num not in content_by_page: content_by_page[page_num] = {'text': None, 'tables': [], 'text_summary': None}
            content_by_page[page_num]['text'] = text
            content_by_page[page_num]['text_summary'] = translated_content['text_summaries'].get(page_num)

        table_summaries_map = translated_content['table_summaries'] # page_idx -> summary
        for page_num, df, idx in translated_content['translated_tables']:
            if page_num not in content_by_page: content_by_page[page_num] = {'text': None, 'tables': [], 'text_summary': None}
            content_by_page[page_num]['tables'].append({'df': df, 'summary': table_summaries_map.get(f"{page_num}_{idx}")})

        # Write sorted content
        for page_num in sorted(content_by_page.keys()):
            data = content_by_page[page_num]
            output_io.write(f"\n\n{'=' * 40}\nPAGE {page_num}\n{'=' * 40}\n\n")

            if data['text_summary']:
                output_io.write("ਪੰਨੇ ਦਾ ਸੰਖੇਪ ਸਾਰ (Page Summary):\n" + "-" * 40 + "\n")
                output_io.write(data['text_summary'] + "\n\n")

            output_io.write(data['text'] if data['text'] else "[No text extracted or translated for this page]\n\n")

            if data['tables']:
                 output_io.write("-" * 40 + "\nTABLES ON THIS PAGE\n" + "-" * 40 + "\n\n")
                 for i, table_data in enumerate(data['tables']):
                     output_io.write(f"Table {i+1}:\n")
                     if table_data['summary']:
                          output_io.write("ਸਾਰਣੀ ਦਾ ਸੰਖੇਪ ਸਾਰ (Table Summary):\n")
                          output_io.write(table_data['summary'] + "\n\n")
                     try:
                         output_io.write(table_data['df'].to_string(index=False, na_rep='-'))
                     except Exception as df_err:
                         output_io.write(f"[Error converting table to string: {df_err}]")
                     output_io.write("\n\n")

        output_io.seek(0)
        return output_io.getvalue().encode('utf-8')
    except Exception as e:
        st.error(f"❌ Error creating TXT output: {e}")
        return b"Error creating TXT content."

def create_punjabi_pdf_bytes_pdf(translated_content, fonts):
    """Create PDF output bytes (specific for this module)."""
    try:
        output_io = io.BytesIO()
        doc = SimpleDocTemplate(output_io, pagesize=A4, topMargin=50, bottomMargin=50, leftMargin=50, rightMargin=50)
        styles = getSampleStyleSheet()

        # Define styles using loaded fonts
        punjabi_style = ParagraphStyle('PunjabiBody', fontName=fonts["regular"]["name"], fontSize=10.5, leading=14, spaceAfter=6)
        punjabi_heading = ParagraphStyle('PunjabiHeading', fontName=fonts["bold"]["name"], fontSize=13, leading=16, spaceAfter=8, spaceBefore=12)
        punjabi_subheading = ParagraphStyle('PunjabiSubHeading', fontName=fonts["bold"]["name"], fontSize=11, leading=14, spaceAfter=4, spaceBefore=8)
        summary_style = ParagraphStyle('SummaryStyle', parent=punjabi_style, backColor=colors.lightyellow, borderPadding=5, borderColor=colors.grey, borderWidth=0.5)
        table_header_style = ParagraphStyle('TableHeader', fontName=fonts["bold"]["name"], fontSize=9.5, alignment=1) # Center align
        table_body_style = ParagraphStyle('TableBody', fontName=fonts["regular"]["name"], fontSize=9)

        elements = []
        elements.append(Paragraph("ਅੰਗਰੇਜ਼ੀ ਤੋਂ ਪੰਜਾਬੀ ਅਨੁਵਾਦਿਤ PDF", punjabi_heading))
        elements.append(Spacer(1, 20))

        # Combine content by page (similar to TXT generation)
        content_by_page = {}
        for page_num, text in translated_content['translated_texts']:
             if page_num not in content_by_page: content_by_page[page_num] = {'text': None, 'tables': [], 'text_summary': None}
             content_by_page[page_num]['text'] = text
             content_by_page[page_num]['text_summary'] = translated_content['text_summaries'].get(page_num)
        table_summaries_map = translated_content['table_summaries']
        for page_num, df, idx in translated_content['translated_tables']:
             if page_num not in content_by_page: content_by_page[page_num] = {'text': None, 'tables': [], 'text_summary': None}
             content_by_page[page_num]['tables'].append({'df': df, 'summary': table_summaries_map.get(f"{page_num}_{idx}")})

        # Build PDF content page by page
        for page_num in sorted(content_by_page.keys()):
            data = content_by_page[page_num]
            elements.append(Paragraph(f"======== ਪੰਨਾ / Page {page_num} ========", punjabi_heading))
            elements.append(Spacer(1, 10))

            if data['text_summary']:
                elements.append(Paragraph("ਪੰਨੇ ਦਾ ਸੰਖੇਪ ਸਾਰ:", punjabi_subheading))
                elements.append(Paragraph(data['text_summary'].replace('\n', '<br/>'), summary_style))
                elements.append(Spacer(1, 10))

            text = data.get('text')
            if text:
                # Split text and format paragraphs
                text_paragraphs = text.split('\n\n')
                for para in text_paragraphs:
                    if para.strip():
                         # Basic list detection (doesn't indent properly without more complex logic)
                         is_list_item = re.match(r'^\s*[-*•\d+\.]+\s+', para)
                         style = punjabi_style
                         # Replace multiple newlines within a paragraph with <br/>
                         para_html = re.sub(r'\n{2,}', '<br/><br/>', para.strip()).replace('\n', '<br/>')
                         elements.append(Paragraph(para_html, style))
                         # Add less space after list items
                         # elements.append(Spacer(1, 2 if is_list_item else 6))
            else:
                elements.append(Paragraph("[ਇਸ ਪੰਨੇ ਲਈ ਕੋਈ ਟੈਕਸਟ ਐਕਸਟਰੈਕਟ ਜਾਂ ਅਨੁਵਾਦ ਨਹੀਂ ਕੀਤਾ ਗਿਆ]", punjabi_style))

            if data['tables']:
                elements.append(Spacer(1, 15))
                elements.append(Paragraph("ਸਾਰਣੀਆਂ / Tables:", punjabi_subheading))
                for i, table_data in enumerate(data['tables']):
                    elements.append(Spacer(1, 8))
                    elements.append(Paragraph(f"ਸਾਰਣੀ / Table {i+1}", punjabi_style))
                    if table_data['summary']:
                        elements.append(Paragraph("ਸੰਖੇਪ ਸਾਰ:", punjabi_subheading))
                        elements.append(Paragraph(table_data['summary'].replace('\n', '<br/>'), summary_style))
                        elements.append(Spacer(1, 5))

                    df = table_data['df']
                    try:
                        # Prepare data for ReportLab table - convert all to string, handle None
                        header = [Paragraph(str(col) if col is not None else '', table_header_style) for col in df.columns]
                        table_data_list = [header]
                        for row in df.itertuples(index=False):
                            table_data_list.append([Paragraph(str(cell) if cell is not None else '', table_body_style) for cell in row])

                        # Calculate dynamic column widths (basic heuristic)
                        num_cols = len(df.columns)
                        page_width = A4[0] - 100 # Page width minus margins
                        col_width = page_width / num_cols if num_cols > 0 else page_width
                        col_widths = [col_width] * num_cols

                        table = Table(table_data_list, colWidths=col_widths, hAlign='LEFT')
                        table.setStyle(TableStyle([
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 3),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                            ('TOPPADDING', (0, 0), (-1, -1), 3),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                        ]))
                        elements.append(table)
                    except Exception as pdf_table_err:
                        st.warning(f"Page {page_num}: Could not render table {i+1} in PDF: {pdf_table_err}")
                        elements.append(Paragraph(f"[Error rendering table {i+1}: {pdf_table_err}]", summary_style)) # Show error in PDF

        # Build the PDF
        doc.build(elements)
        output_io.seek(0)
        return output_io.getvalue()
    except Exception as e:
        st.error(f"❌ Error creating PDF output: {e}")
        st.exception(e)
        # Create a minimal error PDF
        buffer = io.BytesIO()
        p = SimpleDocTemplate(buffer, pagesize=A4)
        p.build([Paragraph(f"Error creating PDF: {e}", getSampleStyleSheet()['Normal'])])
        buffer.seek(0)
        return buffer.getvalue()


# --- Main Function to run the process ---
def run_pdf_translation_process(pdf_bytes, gemini_model, punjabi_fonts, save_intermediate, max_retries):
    """Runs the full En->Pa PDF translation and returns output bytes."""
    start_time = datetime.now()

    # Create output dirs if needed
    os.makedirs(OUTPUT_DIR_PA, exist_ok=True)
    if save_intermediate:
        os.makedirs(CACHE_DIR_PA, exist_ok=True)
        os.makedirs(PROGRESS_DIR_PA, exist_ok=True)

    with st.status("Running English -> Punjabi PDF Translation...", expanded=True) as status:
        st.write("➡️ Step 1: Extracting Text and Tables...")
        extracted_data = extract_tables_and_text_pdf(pdf_bytes)
        if not extracted_data['page_texts'] and not extracted_data['tables']:
             status.update(label="Extraction Failed!", state="error", expanded=True)
             st.error("Could not extract any content from the PDF.")
             return None, None # Return None for both outputs
        st.success("✅ Extraction Complete.")

        st.write("➡️ Step 2: Translating Content to Punjabi...")
        translated_content = translate_content_pdf(
            extracted_data['page_texts'],
            extracted_data['tables'],
            gemini_model,
            save_intermediate=save_intermediate,
            max_retries=max_retries
        )
        st.success("✅ Translation Complete.")

        st.write("➡️ Step 3: Generating Output Files...")
        pdf_output_bytes = create_punjabi_pdf_bytes_pdf(translated_content, punjabi_fonts)
        txt_output_bytes = create_txt_output_bytes_pdf(translated_content)
        st.success("✅ Output Files Generated.")

        status.update(label="En->Pa Translation Completed!", state="complete", expanded=False)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    st.info(f"⏱️ En->Pa Translation finished in {elapsed_time.total_seconds():.2f} seconds.")

    return pdf_output_bytes, txt_output_bytes