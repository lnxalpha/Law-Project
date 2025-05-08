import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance
import zipfile
import tempfile
import os
import PyPDF2
import docx
import re
import nltk
from difflib import SequenceMatcher
from jinja2 import Environment
import cv2
import numpy as np

# Check if the 'punkt' resource is downloaded, if not, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_image(image):
    """Enhance image quality for better OCR results"""
    # Convert to grayscale
    img = image.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Convert to numpy array for OpenCV processing
    img_np = np.array(img)

    # Apply adaptive thresholding
    img_np = cv2.adaptiveThreshold(
        img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to PIL Image
    return Image.fromarray(img_np)

def ocr_images(input_dir):
    pages = []
    # This DEFINITELY works - using triple quotes:
    custom_config = r'''--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}<>-/\'\"@#$%^&*+=|\\_~ "'''

    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            path = os.path.join(input_dir, fname)
            img = Image.open(path)
            processed_img = preprocess_image(img)
            text = pytesseract.image_to_string(
                processed_img,
                config=custom_config,
                lang='eng'
            )
            pages.append(clean_ocr_text(text))
    return '\n'.join(pages)

def clean_ocr_text(text):
    """Clean common OCR errors"""
    # Fix common misread characters
    replacements = {
        '|': 'I', '[': 'I', ']': 'I',
        '©': 'e', '¢': 'c', '®': 'a',
        '‘': "'", '‚': ',', '“': '"',
        '”': '"', '™': '™', '˜': '~'
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # Remove hyphenation at line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'\s+([)])', r'\1', text)

    return text

def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfFileReader(open(pdf_path, 'rb'))
    texts = []
    for i in range(reader.getNumPages()):
        page = reader.getPage(i)
        text = page.extract_text()
        if text:
            texts.append(text)
    return '\n'.join(texts)

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

def normalize_text(text):
    """Normalize text for comparison while preserving line numbers"""
    # Preserve newlines for line number tracking
    lines = text.split('\n')
    normalized_lines = []

    for line in lines:
        if not line.strip():
            normalized_lines.append('')
            continue

        # Basic normalization (preserve case for better matching)
        line = line.replace("–", "-").replace('“', '"').replace('”', '"')
        line = line.replace("‘", "'").replace("’", "'").replace("''", '"')

        # Remove special characters but keep basic punctuation
        line = re.sub(r"[^\w\s.,;:!?()\"'\-]", "", line)

        # Normalize whitespace (but keep single newlines)
        line = re.sub(r"\s+", " ", line).strip()
        normalized_lines.append(line)

    return '\n'.join(normalized_lines)

def split_sentences_with_lines(text):
    """Split text into sentences while preserving line numbers"""
    lines = text.split('\n')
    sentences = []

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue

        # Tokenize sentences within each line
        line_sents = nltk.sent_tokenize(line)
        for sent in line_sents:
            sentences.append({
                'text': sent,
                'line': line_num,
                'original': line  # Keep original for context
            })

    return sentences

def align_texts(source_items, target_items, threshold=0.85):
    """Align texts with improved matching"""
    aligned = []

    for src_item in source_items:
        best_match = None
        best_score = 0

        for tgt_item in target_items:
            # Use both direct comparison and set comparison
            seq_score = SequenceMatcher(None, src_item['text'], tgt_item['text']).ratio()
            set_score = len(set(src_item['text'].split()) & set(tgt_item['text'].split())) / \
                       max(len(set(src_item['text'].split())), 1)

            # Combined score (weighted toward sequence match)
            score = 0.7 * seq_score + 0.3 * set_score

            if score > best_score:
                best_score = score
                best_match = tgt_item

        if best_score < threshold:
            aligned.append({
                'source_text': src_item['text'],
                'target_text': best_match['text'] if best_match else '',
                'source_line': src_item['line'],
                'target_line': best_match['line'] if best_match else 0,
                'score': best_score,
                'source_context': src_item['original'],
                'target_context': best_match['original'] if best_match else ''
            })

    return aligned

def generate_html_report(mismatches):
    env = Environment()
    template_str = '''
    <html><head><meta charset="utf-8"><title>Proofing Report</title>
    <style>
    body { font-family: sans-serif; padding: 20px; }
    .mismatch {
        margin-bottom: 30px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        background: #f9f9f9;
    }
    .header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .score {
        font-weight: bold;
        color: #d9534f;
    }
    .line-info {
        font-size: 0.85em;
        color: #666;
        margin-bottom: 8px;
    }
    .source, .target {
        padding: 10px;
        margin-bottom: 5px;
    }
    .source {
        background: #fdd;
        border-left: 4px solid #d9534f;
    }
    .target {
        background: #dfd;
        border-left: 4px solid #5cb85c;
    }
    .context {
        font-size: 0.8em;
        color: #888;
        margin-top: 5px;
        font-style: italic;
    }
    h1 { color: #333; }
    </style></head><body>
    <h1>Document Comparison Report</h1>
    {% for item in mismatches %}
    <div class="mismatch">
        <div class="header">
            <div class="line-info">
                <strong>Scanned Document:</strong> Line {{ item.source_line }}
                {% if item.target_line %} |
                <strong>Digital Document:</strong> Line {{ item.target_line }}
                {% endif %}
            </div>
            <div class="score">Match: {{ (item.score*100)|round(1) }}%</div>
        </div>

        <div class="source">
            <strong>Scanned Text:</strong><br>
            {{ item.source_text }}
            <div class="context">Full line: {{ item.source_context }}</div>
        </div>

        <div class="target">
            <strong>Digital Text:</strong><br>
            {% if item.target_text %}
                {{ item.target_text }}
                <div class="context">Full line: {{ item.target_context }}</div>
            {% else %}
                No matching text found
            {% endif %}
        </div>
    </div>
    {% endfor %}
    </body></html>
    '''
    template = env.from_string(template_str)
    return template.render(mismatches=mismatches)

# Streamlit App
st.title("Legal Document Comparison Tool")
st.write("Upload scanned images (ZIP or individual files) and a digital reference file (PDF or DOCX).")

# File Uploads
zip_file = st.file_uploader("Upload ZIP of scanned images", type=['zip'])
image_files = st.file_uploader("Or upload individual scanned images", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)
digital_file = st.file_uploader("Upload digital reference (PDF or DOCX)", type=['pdf', 'docx'])

if (zip_file or image_files) and digital_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Handle scanned images
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
        elif image_files:
            for uploaded in image_files:
                img_path = os.path.join(tmpdir, uploaded.name)
                with open(img_path, 'wb') as f:
                    f.write(uploaded.getbuffer())

        # Handle digital file
        digital_path = os.path.join(tmpdir, digital_file.name)
        with open(digital_path, 'wb') as f:
            f.write(digital_file.getbuffer())

        st.info("Processing documents... This may take a few minutes.")

        try:
            # OCR scanned images with improved processing
            scanned_text = ocr_images(tmpdir)

            # Extract from digital source
            if digital_file.name.lower().endswith('.pdf'):
                digital_text = extract_text_from_pdf(digital_path)
            elif digital_file.name.lower().endswith('.docx'):
                digital_text = extract_text_from_docx(digital_path)
            else:
                raise ValueError("Unsupported digital file type.")

            # Normalize while preserving line numbers
            scanned_norm = normalize_text(scanned_text)
            digital_norm = normalize_text(digital_text)

            # Split into sentences with line numbers
            scanned_items = split_sentences_with_lines(scanned_norm)
            digital_items = split_sentences_with_lines(digital_norm)

            # Align with improved matching
            mismatches = align_texts(scanned_items, digital_items)

            if mismatches:
                st.warning(f"Found {len(mismatches)} potential discrepancies")

                # Show summary stats
                avg_score = sum(m['score'] for m in mismatches) / len(mismatches)
                st.write(f"Average match score: {avg_score*100:.1f}%")

                # Generate enhanced report
                html_report = generate_html_report(mismatches)
                st.components.v1.html(html_report, height=800, scrolling=True)

                # Download options
                st.download_button(
                    "Download Full Report (HTML)",
                    html_report,
                    file_name="document_comparison_report.html",
                    mime="text/html"
                )

                # Option to download cleaned OCR text
                st.download_button(
                    "Download Cleaned OCR Text",
                    scanned_text,
                    file_name="cleaned_ocr_text.txt",
                    mime="text/plain"
                )
            else:
                st.success("✅ No significant discrepancies found. Documents match closely.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure your documents are clear and try again.")
