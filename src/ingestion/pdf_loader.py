"""
src/ingestion/pdf_loader.py
------------------------------------------
Extracts text from the IFAB Laws of the Game PDF.
Uses pdfplumber for layout-aware extraction.
Tags each page with Law number metadata (Law 1–17).
"""

import re
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


# Maps common section titles to Law numbers
LAW_PATTERNS = {
    r"law\s+1[\s:\-–]": "Law 1 - The Field of Play",
    r"law\s+2[\s:\-–]": "Law 2 - The Ball",
    r"law\s+3[\s:\-–]": "Law 3 - The Players",
    r"law\s+4[\s:\-–]": "Law 4 - The Players' Equipment",
    r"law\s+5[\s:\-–]": "Law 5 - The Referee",
    r"law\s+6[\s:\-–]": "Law 6 - The Other Match Officials",
    r"law\s+7[\s:\-–]": "Law 7 - The Duration of the Match",
    r"law\s+8[\s:\-–]": "Law 8 - The Start and Restart of Play",
    r"law\s+9[\s:\-–]": "Law 9 - The Ball In and Out of Play",
    r"law\s+10[\s:\-–]": "Law 10 - Determining the Outcome of a Match",
    r"law\s+11[\s:\-–]": "Law 11 - Offside",
    r"law\s+12[\s:\-–]": "Law 12 - Fouls and Misconduct",
    r"law\s+13[\s:\-–]": "Law 13 - Free Kicks",
    r"law\s+14[\s:\-–]": "Law 14 - The Penalty Kick",
    r"law\s+15[\s:\-–]": "Law 15 - The Throw-In",
    r"law\s+16[\s:\-–]": "Law 16 - The Goal Kick",
    r"law\s+17[\s:\-–]": "Law 17 - The Corner Kick",
}


def detect_law_number(text: str) -> str:
    """Detect which Law of the Game a text chunk belongs to."""
    text_lower = text.lower()
    for pattern, law_name in LAW_PATTERNS.items():
        if re.search(pattern, text_lower):
            return law_name
    return "General / Introduction"


def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Load PDF and return list of page dicts with text + metadata.
    
    Returns:
        List of dicts: {
            'page_num': int,
            'text': str,
            'law': str,        # detected Law number
            'source': str,     # filename
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Run: python scripts/download_pdf.py"
        )

    print(f"Loading PDF: {path.name}")
    pages = []
    current_law = "General / Introduction"

    with pdfplumber.open(pdf_path) as pdf:
        print(f"   Total pages: {len(pdf.pages)}")

        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)

            if not text or len(text.strip()) < 30:
                continue  # Skip near-empty pages (covers, dividers)

            # Clean up common PDF artifacts
            text = clean_text(text)

            # Update current Law context if found on this page
            detected = detect_law_number(text)
            if detected != "General / Introduction":
                current_law = detected

            pages.append({
                "page_num": i + 1,
                "text": text,
                "law": current_law,
                "source": path.name,
            })

    print(f"Extracted {len(pages)} usable pages")
    return pages


def clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove page numbers standing alone on a line
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Remove header/footer repetitions
    text = re.sub(r'Laws of the Game 2025/26\s*', '', text)
    return text.strip()
