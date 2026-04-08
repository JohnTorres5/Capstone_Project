import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class CourseArtifacts:
   course: str
   course_dir: Path
   index: faiss.Index
   metadata: List[Dict[str, Any]]
   model_name: str


def _read_json(file_path: Path) -> Any:
   with open(file_path, "r", encoding="utf-8") as file_handle:
      return json.load(file_handle)


def _load_course_artifacts(course_dir: Path) -> Optional[CourseArtifacts]:
   embeddings_dir = course_dir / "embeddings"
   index_path = embeddings_dir / "index.faiss"
   metadata_path = embeddings_dir / "metadata.json"
   config_path = embeddings_dir / "config.json"

   if not index_path.exists() or not metadata_path.exists():
      return None

   index = faiss.read_index(str(index_path))
   metadata_payload = _read_json(metadata_path)
   if not isinstance(metadata_payload, list):
      return None

   model_name = DEFAULT_RETRIEVER_MODEL
   if config_path.exists():
      config_payload = _read_json(config_path)
      if isinstance(config_payload, dict):
         model_name = config_payload.get("model_name", model_name)

   return CourseArtifacts(
      course=course_dir.name,
      course_dir=course_dir,
      index=index,
      metadata=metadata_payload,
      model_name=model_name,
   )


def load_retrieval_corpus(processed_dir: Path, course: Optional[str] = None) -> List[CourseArtifacts]:
   if not processed_dir.exists():
      return []

   if course:
      course_dirs = [processed_dir / course]
   else:
      course_dirs = [path for path in sorted(processed_dir.iterdir()) if path.is_dir()]

   artifacts: List[CourseArtifacts] = []
   for course_dir in course_dirs:
      course_artifacts = _load_course_artifacts(course_dir)
      if course_artifacts is not None:
         artifacts.append(course_artifacts)

   return artifacts


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
   normalized = vector.astype(np.float32, copy=True)
   faiss.normalize_L2(normalized)
   return normalized


def _encode_query(query: str, model_name: str) -> np.ndarray:
   model = SentenceTransformer(model_name)
   query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
   return _normalize_vector(query_vector)


def retrieve_relevant_chunks(
   query: str,
   processed_dir: Path = Path("data/processed"),
   course: Optional[str] = None,
   embedding_model: str = DEFAULT_RETRIEVER_MODEL,
   top_k: int = 5,
) -> List[Dict[str, Any]]:
   artifacts = load_retrieval_corpus(processed_dir=processed_dir, course=course)
   if not artifacts:
      return []

   query_vector = _encode_query(query, embedding_model)
   results: List[Dict[str, Any]] = []

   for course_artifacts in artifacts:
      if course_artifacts.index.ntotal == 0 or not course_artifacts.metadata:
         continue

      search_k = min(top_k, course_artifacts.index.ntotal)
      distances, indices = course_artifacts.index.search(query_vector, search_k)

      for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
         if idx < 0 or idx >= len(course_artifacts.metadata):
            continue

         chunk = dict(course_artifacts.metadata[idx])
         chunk["score"] = float(score)
         chunk["rank"] = rank
         chunk["course"] = chunk.get("course") or course_artifacts.course
         chunk["source_course"] = course_artifacts.course
         results.append(chunk)

   results.sort(key=lambda item: item.get("score", float("-inf")), reverse=True)
   return results[:top_k]


def _format_chunk_location(chunk: Dict[str, Any]) -> str:
   parts: List[str] = []
   course = chunk.get("course")
   source_file = chunk.get("source_file")
   page = chunk.get("page")
   slide = chunk.get("slide")

   if course:
      parts.append(str(course))
   if source_file:
      parts.append(str(source_file))
   if page is not None:
      parts.append(f"page {page}")
   if slide is not None:
      parts.append(f"slide {slide}")

   return " | ".join(parts) if parts else "unknown source"


def build_context(chunks: List[Dict[str, Any]], max_chars_per_chunk: int = 900) -> str:
   sections: List[str] = []

   for position, chunk in enumerate(chunks, start=1):
      excerpt = (chunk.get("text") or "").strip()
      if len(excerpt) > max_chars_per_chunk:
         excerpt = excerpt[:max_chars_per_chunk].rstrip() + "..."

      sections.append(
         f"[{position}] {_format_chunk_location(chunk)}\n"
         f"Score: {chunk.get('score', 0.0):.4f}\n"
         f"{excerpt}"
      )

   return "\n\n".join(sections)


def _load_generator(model_name: str):
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

   model_kwargs = {"trust_remote_code": True}
   if torch.cuda.is_available():
      model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
   else:
      model_kwargs.update({"torch_dtype": torch.float32})

   model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
   model.eval()
   return tokenizer, model


def generate_answer(
   question: str,
   chunks: List[Dict[str, Any]],
   generator_model: str = DEFAULT_GENERATOR_MODEL,
   max_new_tokens: int = 256,
   temperature: float = 0.2,
   top_p: float = 0.9,
) -> str:
   if not chunks:
      return "I could not find relevant course material to answer that question."

   tokenizer, model = _load_generator(generator_model)
   context = build_context(chunks)

   messages = [
      {
         "role": "system",
         "content": (
            "You are an academic study assistant. Answer only using the provided context. "
            "If the context does not contain the answer, say so clearly. "
            "Cite relevant sources using bracketed numbers like [1], [2]."
         ),
      },
      {
         "role": "user",
         "content": (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Write a concise, grounded answer with citations."
         ),
      },
   ]

   if hasattr(tokenizer, "apply_chat_template"):
      prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   else:
      prompt_text = (
         f"System: {messages[0]['content']}\n\n"
         f"User: {messages[1]['content']}\n\nAssistant:"
      )

   inputs = tokenizer(prompt_text, return_tensors="pt")
   if torch.cuda.is_available():
      inputs = {key: value.to(model.device) for key, value in inputs.items()}

   do_sample = temperature > 0
   generation_kwargs = {
      "max_new_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "do_sample": do_sample,
      "pad_token_id": tokenizer.pad_token_id,
      "eos_token_id": tokenizer.eos_token_id,
   }

   with torch.inference_mode():
      output_ids = model.generate(**inputs, **generation_kwargs)

   generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
   answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
   return answer or "The model did not produce a usable answer."


def run_rag_pipeline(
   question: str,
   processed_dir: Path = Path("data/processed"),
   course: Optional[str] = None,
   embedding_model: str = DEFAULT_RETRIEVER_MODEL,
   top_k: int = 5,
   generator_model: str = DEFAULT_GENERATOR_MODEL,
   max_new_tokens: int = 256,
   temperature: float = 0.2,
) -> Dict[str, Any]:
   retrieved_chunks = retrieve_relevant_chunks(
      query=question,
      processed_dir=processed_dir,
      course=course,
      embedding_model=embedding_model,
      top_k=top_k,
   )
   answer = generate_answer(
      question=question,
      chunks=retrieved_chunks,
      generator_model=generator_model,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
   )

   return {
      "question": question,
      "answer": answer,
      "retrieved_chunks": retrieved_chunks,
      "embedding_model": embedding_model,
      "generator_model": generator_model,
      "course": course,
      "top_k": top_k,
   }


def print_rag_result(result: Dict[str, Any]) -> None:
   print("Question:")
   print(result.get("question", ""))
   print()
   print("Answer:")
   print(result.get("answer", ""))
   print()
   print("Sources:")
   for idx, chunk in enumerate(result.get("retrieved_chunks", []), start=1):
      location = _format_chunk_location(chunk)
      score = chunk.get("score", 0.0)
      preview = (chunk.get("text") or "").strip().replace("\n", " ")
      if len(preview) > 220:
         preview = preview[:220].rstrip() + "..."
      print(f"  [{idx}] {location} | score={score:.4f}")
      print(f"      {preview}")
