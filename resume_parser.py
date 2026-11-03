"""
resume_parser.py - Advanced Resume Parser & Entity Extraction Module.
Extracts and normalizes text from PDF and DOCX documents.
Identifies structured sections and contact information:
- Candidate Name, Email, Phone, Social Profiles (LinkedIn, GitHub)
- Professional Summary / Career Objective
- Education & Degrees
- Technical Skills (Languages, Frameworks, Databases, Tools)
- Work Experience & Internships
- Projects & Academic Work
- Certifications & Achievements
"""

import io
import re
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

# Extraction thresholds
MIN_DIGITAL_TEXT_THRESHOLD = 50
MAX_ALLOWED_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

# Common technical vocabulary for categorization
COMMON_LANGUAGES = {
    "python", "javascript", "typescript", "java", "c++", "c#", "c", "ruby",
    "go", "golang", "rust", "php", "swift", "kotlin", "r", "scala", "dart", "html", "css", "sql"
}
COMMON_FRAMEWORKS = {
    "react", "react.js", "reactjs", "angular", "vue", "vue.js", "next.js", "nextjs",
    "flask", "django", "fastapi", "express", "express.js", "node.js", "nodejs",
    "spring", "spring boot", "asp.net", "laravel", "flutter", "tensorflow", "pytorch",
    "scikit-learn", "keras", "pandas", "numpy"
}
COMMON_DATABASES = {
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis",
    "cassandra", "oracle", "mariadb", "dynamodb", "elasticsearch", "firebase"
}
COMMON_TOOLS = {
    "git", "github", "gitlab", "docker", "kubernetes", "k8s", "aws", "azure",
    "gcp", "google cloud", "linux", "jenkins", "terraform", "ansible", "postman", "jira"
}


def clean_extracted_text(text: str) -> str:
    """Cleans up excess whitespace, control characters, and line breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t"))
    return text.strip()


def extract_contact_info(text: str) -> Dict[str, Any]:
    """Extracts candidate contact details from resume text."""
    contact = {
        "name": None,
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None
    }

    # Extract Email
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if email_match:
        contact["email"] = email_match.group(0).strip()

    # Extract Phone (North American & International formats)
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    if phone_match:
        contact["phone"] = phone_match.group(0).strip()

    # Extract LinkedIn / GitHub
    li_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    if li_match:
        contact["linkedin"] = li_match.group(0)

    gh_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    if gh_match:
        contact["github"] = gh_match.group(0)

    # Extract Candidate Name heuristic (first 1-3 lines excluding headings and emails)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    invalid_name_words = {
        "resume", "curriculum", "vitae", "cv", "page", "phone", "email",
        "contact", "address", "education", "experience", "skills", "profile"
    }

    for line in lines[:5]:
        cleaned_line = re.sub(r"[^a-zA-Z\s]", "", line).strip()
        words = cleaned_line.split()
        if 2 <= len(words) <= 4:
            if not any(w.lower() in invalid_name_words for w in words):
                contact["name"] = cleaned_line
                break

    return contact


def segment_resume_sections(text: str) -> Dict[str, str]:
    """Segments the resume text into standard recognized headings."""
    section_patterns = {
        "summary": r"(?:professional\s+summary|career\s+summary|career\s+objective|summary|profile|about\s+me)",
        "skills": r"(?:technical\s+skills|skills\s+and\s+competencies|core\s+competencies|skills|technologies|technical\s+proficiencies)",
        "experience": r"(?:work\s+experience|professional\s+experience|employment\s+history|experience|internships|work\s+history)",
        "projects": r"(?:projects|academic\s+projects|personal\s+projects|key\s+projects)",
        "education": r"(?:education|academic\s+background|educational\s+qualifications|qualifications|academic\s+history)",
        "certifications": r"(?:certifications|licenses\s+and\s+certifications|certificates|professional\s+certifications)",
        "achievements": r"(?:achievements|awards\s+and\s+honors|honors|accomplishments)"
    }

    # Build regex to find headers
    header_regex = re.compile(
        r"(?m)^(?:[0-9\.\-\*#\s]*)(?P<header>" + "|".join(section_patterns.values()) + r")[:\s]*$",
        re.IGNORECASE
    )

    matches = list(header_regex.finditer(text))
    sections = {}

    if not matches:
        sections["full_content"] = text
        return sections

    # Capture content before the first header (usually header/contact)
    first_start = matches[0].start()
    if first_start > 0:
        sections["header_info"] = text[:first_start].strip()

    for i, match in enumerate(matches):
        matched_text = match.group("header").lower()
        # Find which section category this belongs to
        category = "other"
        for cat, pattern in section_patterns.items():
            if re.search(pattern, matched_text, re.IGNORECASE):
                category = cat
                break

        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[start_pos:end_pos].strip()

        if category in sections:
            sections[category] += "\n\n" + section_content
        else:
            sections[category] = section_content

    return sections


def categorize_skills(text: str) -> Dict[str, List[str]]:
    """Identifies and groups recognized skills into languages, frameworks, databases, and tools."""
    lower_text = text.lower()
    # Tokenize words and common combinations
    tokens = set(re.findall(r"[a-zA-Z0-9#\+\.-]+", lower_text))

    detected_langs = [l.capitalize() for l in sorted(COMMON_LANGUAGES) if l in tokens or f" {l} " in f" {lower_text} "]
    detected_frameworks = [f.title() for f in sorted(COMMON_FRAMEWORKS) if f in tokens or f" {f} " in f" {lower_text} "]
    detected_dbs = [d.title() for d in sorted(COMMON_DATABASES) if d in tokens or f" {d} " in f" {lower_text} "]
    detected_tools = [t.title() for t in sorted(COMMON_TOOLS) if t in tokens or f" {t} " in f" {lower_text} "]

    return {
        "programming_languages": detected_langs,
        "frameworks_and_libraries": detected_frameworks,
        "databases": detected_dbs,
        "cloud_and_tools": detected_tools
    }


def extract_text_from_pdf(pdf_stream: io.BytesIO) -> Tuple[str, str]:
    """
    Extracts text from a PDF input stream.
    First tries digital extraction with pypdf.
    If extracted text is under MIN_DIGITAL_TEXT_THRESHOLD, attempts OCR via Tesseract.
    """
    extracted_text = ""
    method = "pypdf"

    try:
        from pypdf import PdfReader
        pdf_stream.seek(0)
        reader = PdfReader(pdf_stream)
        text_parts = []
        for page in reader.pages:
            pt = page.extract_text() or ""
            if pt.strip():
                text_parts.append(pt.strip())

        extracted_text = clean_extracted_text("\n\n".join(text_parts))
    except Exception as e:
        logger.warning(f"pypdf extraction failed or encountered error: {e}")
        extracted_text = ""

    # Check if digital extraction succeeded with sufficient content
    if len(extracted_text) >= MIN_DIGITAL_TEXT_THRESHOLD:
        return extracted_text, "pypdf"

    # Attempt OCR fallback for scanned resumes
    logger.info("PDF has low or empty digital text. Attempting Tesseract OCR fallback...")
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        pdf_stream.seek(0)
        images = convert_from_bytes(pdf_stream.read())
        ocr_parts = []
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                ocr_parts.append(ocr_text.strip())

        full_ocr = clean_extracted_text("\n\n".join(ocr_parts))
        if len(full_ocr) > len(extracted_text):
            return full_ocr, "tesseract_ocr"
        elif extracted_text:
            return extracted_text, "pypdf_low_yield"
        else:
            return "", "empty"

    except Exception as ocr_err:
        logger.warning(f"OCR fallback unavailable or failed: {ocr_err}")
        return extracted_text, "pypdf" if extracted_text else "failed"


def extract_text_from_docx(docx_stream: io.BytesIO) -> Tuple[str, str]:
    """
    Extracts text from a DOCX file stream including paragraphs and tables.
    """
    try:
        import docx
        docx_stream.seek(0)
        doc = docx.Document(docx_stream)
        parts = []

        # Extract standard paragraphs
        for p in doc.paragraphs:
            if p.text.strip():
                parts.append(p.text.strip())

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if row_cells:
                    parts.append(" | ".join(row_cells))

        full_text = clean_extracted_text("\n".join(parts))
        return full_text, "python-docx"

    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        return "", "docx_error"


def parse_resume_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Dispatches file to appropriate parser, extracts text, contact info,
    and structured sections for instant UI preview and analysis.
    """
    if len(file_bytes) > MAX_ALLOWED_FILE_SIZE:
        raise ValueError(f"File size ({len(file_bytes)} bytes) exceeds maximum limit of 16 MB.")

    lower_name = filename.lower()
    stream = io.BytesIO(file_bytes)

    if lower_name.endswith(".pdf"):
        text, method = extract_text_from_pdf(stream)
    elif lower_name.endswith(".docx"):
        text, method = extract_text_from_docx(stream)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Supported formats are .pdf and .docx.")

    if not text or len(text.strip()) < 20:
        raise ValueError("Could not extract sufficient readable text from the document. Please verify the file.")

    # Extract rich preview entities
    contact = extract_contact_info(text)
    sections = segment_resume_sections(text)
    categorized_skills = categorize_skills(text)

    warnings = []
    if method == "tesseract_ocr":
        warnings.append("Document was scanned; text was extracted using Tesseract OCR. Formatting may vary.")
    if not contact.get("email"):
        warnings.append("No email address detected in resume header.")
    if not sections.get("skills"):
        warnings.append("No explicit 'Skills' section heading detected.")

    return {
        "filename": filename,
        "character_count": len(text),
        "word_count": len(text.split()),
        "extraction_method": method,
        "extracted_text": text,
        "contact_info": contact,
        "sections": sections,
        "categorized_skills": categorized_skills,
        "warnings": warnings
    }
