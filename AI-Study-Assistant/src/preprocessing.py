import argparse
from pathlib import Path

from .image_extraction import extract_images_for_course
from .embeddings import run_embedding_generation
from .pdf_to_json import preprocess_course
from .rag_pipeline import print_rag_result, run_rag_pipeline
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


def run_image_extraction() -> None:
    if not RAW_DATA_DIR.exists():
        print("Raw data directory does not exist.")
        return

    for course_folder in RAW_DATA_DIR.iterdir():
        if not course_folder.is_dir():
            continue

        course_name = course_folder.name
        print(f"\nStarting image extraction for course: {course_name}")
        extract_images_for_course(course_dir=course_folder, processed_dir=PROCESSED_DATA_DIR)

    print("\nImage extraction complete.")


def run_pipeline(
    run_pdf: bool = True,
    run_images: bool = True,
    run_chunking: bool = True,
    run_embeddings: bool = True,
    run_rag: bool = True,
    course: str = None,
    chunk_size: int = 550,
    overlap: int = 80,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    overwrite_embeddings: bool = False,
    rag_question: str = None,
    rag_top_k: int = 5,
    rag_generator_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    rag_max_new_tokens: int = 256,
    rag_temperature: float = 0.2,
) -> None:
    if run_pdf:
        run_pdf_to_json()

    if run_images:
        print("\nStarting image extraction...")
        run_image_extraction()

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

    if run_rag:
        if not rag_question:
            raise ValueError("rag_question is required when run_rag=True")

        print("\nStarting RAG query...")
        rag_result = run_rag_pipeline(
            question=rag_question,
            processed_dir=PROCESSED_DATA_DIR,
            course=course,
            embedding_model=embedding_model,
            top_k=rag_top_k,
            generator_model=rag_generator_model,
            max_new_tokens=rag_max_new_tokens,
            temperature=rag_temperature,
        )
        print_rag_result(rag_result)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline steps.")
    parser.add_argument("--course", type=str, default=None, help="Optional course to process")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF/PPTX to JSON stage")
    parser.add_argument("--skip-images", action="store_true", help="Skip image extraction stage")
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
    parser.add_argument("--run-rag", action="store_true", help="Run the RAG query stage after embeddings")
    parser.add_argument("--rag-question", type=str, default=None, help="Question to answer with the RAG pipeline")
    parser.add_argument("--rag-top-k", type=int, default=5, help="Number of retrieved chunks to feed the generator")
    parser.add_argument(
        "--rag-generator-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Causal language model used to generate answers",
    )
    parser.add_argument(
        "--rag-max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate for the answer",
    )
    parser.add_argument(
        "--rag-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for answer generation",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_pipeline(
        run_pdf=not args.skip_pdf,
        run_images=not args.skip_images,
        run_chunking=not args.skip_chunking,
        run_embeddings=not args.skip_embeddings,
        course=args.course,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        overwrite_embeddings=args.overwrite_embeddings,
        run_rag=args.run_rag,
        rag_question=args.rag_question,
        rag_top_k=args.rag_top_k,
        rag_generator_model=args.rag_generator_model,
        rag_max_new_tokens=args.rag_max_new_tokens,
        rag_temperature=args.rag_temperature,
    )
