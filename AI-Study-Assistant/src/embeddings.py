import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _read_json(file_path: Path) -> Any:
   with open(file_path, "r", encoding="utf-8") as f:
      return json.load(f)


def _extract_chunks_from_file(file_path: Path, default_course: str) -> List[Dict[str, Any]]:
   payload = _read_json(file_path)

   if isinstance(payload, list):
      raw_chunks = payload
   elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
      raw_chunks = payload["chunks"]
   else:
      return []

   normalized: List[Dict[str, Any]] = []
   source_file_default = file_path.stem

   for idx, chunk in enumerate(raw_chunks):
      if not isinstance(chunk, dict):
         continue

      text = (
         chunk.get("text")
         or chunk.get("chunk_text")
         or chunk.get("content")
         or ""
      )
      text = text.strip()
      if not text:
         continue

      chunk_record = {
         "chunk_id": chunk.get("chunk_id") or f"{source_file_default}-{idx}",
         "text": text,
         "course": chunk.get("course") or default_course,
         "source_file": chunk.get("source_file")
         or chunk.get("document")
         or source_file_default,
         "page": chunk.get("page"),
         "slide": chunk.get("slide"),
         "chunk_index": chunk.get("chunk_index", idx),
         "metadata": chunk.get("metadata", {}),
      }
      normalized.append(chunk_record)

   return normalized


def load_course_chunks(course_dir: Path, max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
   chunks_dir = course_dir / "chunks"
   if not chunks_dir.exists():
      return []

   course_name = course_dir.name
   all_chunks: List[Dict[str, Any]] = []

   for chunk_file in sorted(chunks_dir.glob("*.json")):
      all_chunks.extend(_extract_chunks_from_file(chunk_file, course_name))
      if max_chunks is not None and len(all_chunks) >= max_chunks:
         return all_chunks[:max_chunks]

   return all_chunks


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
   normalized_vectors = vectors.copy()
   faiss.normalize_L2(normalized_vectors)
   index = faiss.IndexFlatIP(normalized_vectors.shape[1])
   index.add(normalized_vectors)
   return index


def generate_course_embeddings(
   course_dir: Path,
   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
   batch_size: int = 32,
   max_chunks: Optional[int] = None,
   overwrite: bool = False,
) -> Tuple[int, Optional[Path]]:
   embeddings_dir = course_dir / "embeddings"
   embeddings_dir.mkdir(parents=True, exist_ok=True)

   index_path = embeddings_dir / "index.faiss"
   metadata_path = embeddings_dir / "metadata.json"
   config_path = embeddings_dir / "config.json"

   if index_path.exists() and metadata_path.exists() and not overwrite:
      print(f"Skipping {course_dir.name}: embeddings already exist.")
      return 0, embeddings_dir

   chunks = load_course_chunks(course_dir, max_chunks=max_chunks)
   if not chunks:
      print(f"No chunks found for {course_dir.name}. Run text chunking first.")
      return 0, None

   texts = [chunk["text"] for chunk in chunks]

   print(f"Loading embedding model: {model_name}")
   model = SentenceTransformer(model_name)

   print(f"Encoding {len(texts)} chunks for {course_dir.name}...")
   vectors = model.encode(
      texts,
      batch_size=batch_size,
      show_progress_bar=True,
      convert_to_numpy=True,
      normalize_embeddings=False,
   ).astype(np.float32)

   index = build_faiss_index(vectors)

   faiss.write_index(index, str(index_path))
   with open(metadata_path, "w", encoding="utf-8") as f:
      json.dump(chunks, f, indent=2, ensure_ascii=False)

   config = {
      "course": course_dir.name,
      "model_name": model_name,
      "vector_dim": int(vectors.shape[1]),
      "chunk_count": len(chunks),
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "distance_metric": "cosine (via inner product on L2-normalized vectors)",
      "files": {
         "index": index_path.name,
         "metadata": metadata_path.name,
      },
   }
   with open(config_path, "w", encoding="utf-8") as f:
      json.dump(config, f, indent=2)

   print(
      f"Saved embeddings for {course_dir.name}: {len(chunks)} chunks, "
      f"dimension {vectors.shape[1]}"
   )
   return len(chunks), embeddings_dir


def run_embedding_generation(
   processed_dir: Path = Path("data/processed"),
   course: Optional[str] = None,
   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
   batch_size: int = 32,
   max_chunks: Optional[int] = None,
   overwrite: bool = False,
) -> None:
   if not processed_dir.exists():
      print(f"Processed directory not found: {processed_dir}")
      return

   if course:
      course_dirs = [processed_dir / course]
   else:
      course_dirs = [p for p in sorted(processed_dir.iterdir()) if p.is_dir()]

   total_courses = 0
   total_chunks = 0

   for course_dir in course_dirs:
      if not course_dir.exists() or not course_dir.is_dir():
         print(f"Skipping missing course folder: {course_dir.name}")
         continue

      total_courses += 1
      chunk_count, output_dir = generate_course_embeddings(
         course_dir=course_dir,
         model_name=model_name,
         batch_size=batch_size,
         max_chunks=max_chunks,
         overwrite=overwrite,
      )
      if output_dir is not None:
         total_chunks += chunk_count

   print(
      f"Embedding generation complete. Courses processed: {total_courses}, "
      f"total chunks embedded: {total_chunks}"
   )


def _build_arg_parser() -> argparse.ArgumentParser:
   parser = argparse.ArgumentParser(description="Generate FAISS embeddings for course chunks.")
   parser.add_argument(
      "--processed-dir",
      type=Path,
      default=Path("data/processed"),
      help="Directory containing per-course processed folders.",
   )
   parser.add_argument(
      "--course",
      type=str,
      default=None,
      help="Optional course folder to process (e.g., CSC340).",
   )
   parser.add_argument(
      "--model-name",
      type=str,
      default="sentence-transformers/all-MiniLM-L6-v2",
      help="SentenceTransformer model name.",
   )
   parser.add_argument(
      "--batch-size",
      type=int,
      default=32,
      help="Batch size used by SentenceTransformer.encode().",
   )
   parser.add_argument(
      "--max-chunks",
      type=int,
      default=None,
      help="Optional cap for number of chunks per course (for testing).",
   )
   parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing index.faiss and metadata.json if they already exist.",
   )
   return parser


if __name__ == "__main__":
   args = _build_arg_parser().parse_args()
   run_embedding_generation(
      processed_dir=args.processed_dir,
      course=args.course,
      model_name=args.model_name,
      batch_size=args.batch_size,
      max_chunks=args.max_chunks,
      overwrite=args.overwrite,
   )