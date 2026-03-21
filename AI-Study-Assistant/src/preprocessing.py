import os
from pathlib import Path
from pdf_to_json import preprocess_course

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def run_pipeline():
    # Convert documents to JSON format
    if not RAW_DATA_DIR.exists():
        print("Raw data directory does not exist.")
        return

    for course_folder in RAW_DATA_DIR.iterdir():
        if not course_folder.is_dir():
            continue

        course_name = course_folder.name
        print(f"\nStarting preprocessing for course: {course_name}")

        output_json_dir = PROCESSED_DATA_DIR / course_name / "json"

        preprocess_course(
            input_dir=str(course_folder),
            output_dir=str(output_json_dir),
            course_name=course_name
        )

    print("\nPDF → JSON preprocessing complete.")

    # text chunking
    # image extraction
    # embedding generation

    

if __name__ == "__main__":
    run_pipeline()