import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(file_path: Path) -> Dict[str, Any]:
   with open(file_path, "r", encoding="utf-8") as f:
      return json.load(f)


def _token_like_split(text: str) -> List[str]:
   # A simple, fast approximation for token-based chunking.
   return text.split()


def _windowed_chunks(words: List[str], chunk_size: int, overlap: int) -> List[str]:
   if not words:
      return []

   if chunk_size <= 0:
      raise ValueError("chunk_size must be > 0")
   if overlap < 0:
      raise ValueError("overlap must be >= 0")
   if overlap >= chunk_size:
      raise ValueError("overlap must be smaller than chunk_size")

   chunks: List[str] = []
   start = 0
   step = chunk_size - overlap

   while start < len(words):
      end = min(start + chunk_size, len(words))
      chunks.append(" ".join(words[start:end]).strip())
      if end >= len(words):
         break
      start += step

   return chunks


def _chunk_pages(
   pages: List[Dict[str, Any]],
   course: str,
   source_file: str,
   chunk_size: int,
   overlap: int,
) -> List[Dict[str, Any]]:
   chunks: List[Dict[str, Any]] = []
   running_idx = 0

   for item in pages:
      page_text = (item.get("text") or "").strip()
      if not page_text:
         continue

      words = _token_like_split(page_text)
      split_chunks = _windowed_chunks(words, chunk_size=chunk_size, overlap=overlap)

      for local_idx, chunk_text in enumerate(split_chunks):
         chunk_meta = {
            "chunk_id": f"{Path(source_file).stem}-{running_idx}",
            "course": course,
            "source_file": source_file,
            "chunk_index": running_idx,
            "chunk_in_page": local_idx,
            "page": item.get("page"),
            "slide": item.get("slide"),
            "text": chunk_text,
         }
         chunks.append(chunk_meta)
         running_idx += 1

   return chunks


def chunk_document(
   doc_path: Path,
   output_dir: Path,
   chunk_size: int = 550,
   overlap: int = 80,
) -> int:
   doc = _read_json(doc_path)
   course = doc.get("course", "unknown")
   source_file = doc.get("source_file", doc_path.name)
   pages = doc.get("pages", [])
   full_text = (doc.get("text") or "").strip()

   if not isinstance(pages, list):
      pages = []

   chunks = _chunk_pages(
      pages=pages,
      course=course,
      source_file=source_file,
      chunk_size=chunk_size,
      overlap=overlap,
   )

   # Fallback for documents that only have top-level text.
   if not chunks and full_text:
      words = _token_like_split(full_text)
      split_chunks = _windowed_chunks(words, chunk_size=chunk_size, overlap=overlap)
      chunks = [
         {
            "chunk_id": f"{Path(source_file).stem}-{idx}",
            "course": course,
            "source_file": source_file,
            "chunk_index": idx,
            "chunk_in_page": None,
            "page": None,
            "slide": None,
            "text": chunk_text,
         }
         for idx, chunk_text in enumerate(split_chunks)
      ]

   output_dir.mkdir(parents=True, exist_ok=True)
   out_path = output_dir / doc_path.name

   payload = {
      "course": course,
      "source_file": source_file,
      "chunk_size": chunk_size,
      "overlap": overlap,
      "chunk_count": len(chunks),
      "chunks": chunks,
   }

   with open(out_path, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, ensure_ascii=False)

   return len(chunks)


def run_text_chunking(
   processed_dir: Path = Path("data/processed"),
   course: Optional[str] = None,
   chunk_size: int = 550,
   overlap: int = 80,
) -> None:
   if not processed_dir.exists():
      print(f"Processed directory not found: {processed_dir}")
      return

   if course:
      course_dirs = [processed_dir / course]
   else:
      course_dirs = [p for p in sorted(processed_dir.iterdir()) if p.is_dir()]

   total_docs = 0
   total_chunks = 0

   for course_dir in course_dirs:
      if not course_dir.exists() or not course_dir.is_dir():
         print(f"Skipping missing course folder: {course_dir.name}")
         continue

      json_dir = course_dir / "json"
      chunks_dir = course_dir / "chunks"

      if not json_dir.exists():
         print(f"Skipping {course_dir.name}: json directory not found")
         continue

      doc_paths = sorted(json_dir.glob("*.json"))
      if not doc_paths:
         print(f"No JSON files found for {course_dir.name}")
         continue

      print(f"Chunking {len(doc_paths)} documents for {course_dir.name}...")

      for doc_path in doc_paths:
         chunk_count = chunk_document(
            doc_path=doc_path,
            output_dir=chunks_dir,
            chunk_size=chunk_size,
            overlap=overlap,
         )
         total_docs += 1
         total_chunks += chunk_count

      print(f"Finished {course_dir.name}: {len(doc_paths)} docs, {total_chunks} total chunks so far")

   print(
      f"Text chunking complete. Documents processed: {total_docs}, "
      f"chunks generated: {total_chunks}"
   )


def _build_arg_parser() -> argparse.ArgumentParser:
   parser = argparse.ArgumentParser(description="Chunk processed course JSON files.")
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
      "--chunk-size",
      type=int,
      default=550,
      help="Approximate chunk size in word-like tokens.",
   )
   parser.add_argument(
      "--overlap",
      type=int,
      default=80,
      help="Overlapping tokens between chunks.",
   )
   return parser


if __name__ == "__main__":
   args = _build_arg_parser().parse_args()
   run_text_chunking(
      processed_dir=args.processed_dir,
      course=args.course,
      chunk_size=args.chunk_size,
      overlap=args.overlap,
   )