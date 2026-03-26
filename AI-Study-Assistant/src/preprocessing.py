import argparse
from pathlib import Path

from .embeddings import run_embedding_generation
from .pdf_to_json import preprocess_course
from .text_chunking import run_text_chunking

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def run_pdf_to_json() -> None:
    # Convert documents to JSON format.
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
            course_name=course_name,
        )

    print("\nPDF -> JSON preprocessing complete.")


def run_pipeline(
    run_pdf: bool = True,
    run_chunking: bool = True,
    run_embeddings: bool = True,
    course: str = None,
    chunk_size: int = 550,
    overlap: int = 80,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    overwrite_embeddings: bool = False,
) -> None:
    if run_pdf:
        run_pdf_to_json()

    if run_chunking:
        print("\nStarting text chunking...")
        run_text_chunking(
            processed_dir=PROCESSED_DATA_DIR,
            course=course,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if run_embeddings:
        print("\nStarting embedding generation...")
        run_embedding_generation(
            processed_dir=PROCESSED_DATA_DIR,
            course=course,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            overwrite=overwrite_embeddings,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline steps.")
    parser.add_argument("--course", type=str, default=None, help="Optional course to process")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF/PPTX to JSON stage")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip text chunking stage")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation stage")
    parser.add_argument("--chunk-size", type=int, default=550, help="Chunk size in word-like tokens")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap in word-like tokens")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Overwrite existing embedding artifacts",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_pipeline(
        run_pdf=not args.skip_pdf,
        run_chunking=not args.skip_chunking,
        run_embeddings=not args.skip_embeddings,
        course=args.course,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        overwrite_embeddings=args.overwrite_embeddings,
    )
