"""
scripts/download_pdf.py
=======================
Downloads the IFAB Laws of the Game 2025/26 PDF.
Handles redirects and saves locally.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PDF_URL = os.getenv(
    "PDF_URL",
    "https://downloads.theifab.com/downloads/laws-of-the-game-2025-26-double-pages?l=en"
)
PDF_LOCAL_PATH = os.getenv("PDF_LOCAL_PATH", "./data/laws_of_the_game.pdf")


def download_pdf():
    """Download the IFAB Laws of the Game PDF."""
    # Ensure data directory exists
    Path(PDF_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)

    if Path(PDF_LOCAL_PATH).exists():
        size_mb = Path(PDF_LOCAL_PATH).stat().st_size / (1024 * 1024)
        print(f"PDF already exists: {PDF_LOCAL_PATH} ({size_mb:.1f} MB)")
        return PDF_LOCAL_PATH

    print(f"Downloading Laws of the Game PDF from IFAB...")
    print(f"   URL: {PDF_URL}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
    }

    response = requests.get(PDF_URL, headers=headers, stream=True, timeout=60, allow_redirects=True)
    response.raise_for_status()

    # Check we got a PDF
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type and "octet-stream" not in content_type:
        print(f"Warning: Content-Type is '{content_type}'. Saving anyway...")

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(PDF_LOCAL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r   Progress: {pct:.1f}% ({downloaded/(1024*1024):.1f} MB)", end="")

    print(f"\nDownloaded: {PDF_LOCAL_PATH}")
    size_mb = Path(PDF_LOCAL_PATH).stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.1f} MB")
    return PDF_LOCAL_PATH


if __name__ == "__main__":
    download_pdf()
