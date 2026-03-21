import os
import json
import re
from pathlib import Path
import fitz 
from pptx import Presentation

# Text Cleaning 
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = text.replace('\x00', '')  # remove null chars
    return text.strip()

# PDF Extraction
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text() + "\n"

    doc.close()
    return full_text

# PPTX Extraction
def extract_text_from_pptx(file_path: str) -> str:
    presentation = Presentation(file_path)
    full_text = ""

    for slide_number, slide in enumerate(presentation.slides, start=1):
        slide_text = f"\n--- Slide {slide_number} ---\n"
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text += shape.text + "\n"
        full_text += slide_text

    return full_text

# Preprocessing Function
def preprocess_course(input_dir: str, output_dir: str, course_name: str):
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)

    for file_path in input_path.glob("*"):
        if file_path.suffix.lower() not in [".pdf", ".pptx"]:
            continue

        print(f"Processing: {file_path.name}")

        # Extract text
        if file_path.suffix.lower() == ".pdf":
            raw_text = extract_text_from_pdf(str(file_path))
        elif file_path.suffix.lower() == ".pptx":
            raw_text = extract_text_from_pptx(str(file_path))

        # Clean text
        cleaned_text = clean_text(raw_text)

        # Create structured output
        document_data = {
            "course": course_name,
            "source_file": file_path.name,
            "text": cleaned_text
        }

        # Save as JSON
        output_file = Path(output_dir) / f"{file_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(document_data, f, indent=2)

        print(f"Saved: {output_file.name}")

    print(f"\nFinished preprocessing for course: {course_name}")