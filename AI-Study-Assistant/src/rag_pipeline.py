import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
   from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
   Qwen2_5_VLForConditionalGeneration = None

DEFAULT_RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct" 

_EMBEDDER_CACHE: Dict[str, SentenceTransformer] = {}
_GENERATOR_CACHE: Dict[str, Any] = {}

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
   model = _EMBEDDER_CACHE.get(model_name)
   if model is None:
      model = SentenceTransformer(model_name)
      _EMBEDDER_CACHE[model_name] = model
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
   cached = _GENERATOR_CACHE.get(model_name)
   if cached is not None:
      return cached

   model_kwargs = {"trust_remote_code": True}
   if torch.cuda.is_available():
      model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
   else:
      model_kwargs.update({"torch_dtype": torch.float32})

   if "VL" in model_name.upper() and Qwen2_5_VLForConditionalGeneration is not None:
      processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
      model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
      model.eval()
      _GENERATOR_CACHE[model_name] = (processor, model)
      return processor, model

   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

   model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
   model.eval()
   _GENERATOR_CACHE[model_name] = (tokenizer, model)
   return tokenizer, model

# TODO: Add generate_answer_multimodal(question, chunks, image_input, ...)
# TODO: Use the multimodal generator only when image_input is provided.
# TODO: If multimodal generation fails, fall back to generate_answer(...) instead of raising.

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

   model_io, model = _load_generator(generator_model)
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

   if hasattr(model_io, "apply_chat_template"):
      prompt_text = model_io.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   else:
      prompt_text = (
         f"System: {messages[0]['content']}\n\n"
         f"User: {messages[1]['content']}\n\nAssistant:"
      )

   inputs = model_io(text=[prompt_text], return_tensors="pt")
   if torch.cuda.is_available():
      inputs = {key: value.to(model.device) for key, value in inputs.items()}

   do_sample = temperature > 0
   generation_kwargs = {
      "max_new_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "do_sample": do_sample,
      "pad_token_id": getattr(getattr(model_io, "tokenizer", model_io), "pad_token_id", None),
      "eos_token_id": getattr(getattr(model_io, "tokenizer", model_io), "eos_token_id", None),
   }

   with torch.inference_mode():
      output_ids = model.generate(**inputs, **generation_kwargs)

   generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
   tokenizer = getattr(model_io, "tokenizer", model_io)
   answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
   return answer or "The model did not produce a usable answer."


# TODO: Route to generate_answer(...) when image_input is None.
# TODO: Route to generate_answer_multimodal(...) when image_input is provided.
# TODO: Keep the returned mode field in sync with the path that ran.

def run_rag_pipeline(
   question: str,
   processed_dir: Path = Path("data/processed"),
   course: Optional[str] = None,
   image_input: Optional[Any] = None,
   embedding_model: str = DEFAULT_RETRIEVER_MODEL,
   top_k: int = 5,
   generator_model: str = DEFAULT_GENERATOR_MODEL,
   max_new_tokens: int = 256,
   temperature: float = 0.2,
) -> Dict[str, Any]:
   """Run retrieval and answer generation, then return a stable response payload.

   Current behavior:
      This function always uses the text-only generation path.
      image_input is accepted so the caller can record whether the request
      was intended for a future multimodal branch.
   """
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
      "mode": "multimodal" if image_input is not None else "text",
   }


def run_rag_backend(
   question: str,
   course: Optional[str] = None,
   image_input: Optional[Any] = None,
   top_k: int = 5,
   max_new_tokens: int = 256,
   temperature: float = 0.2,
) -> Dict[str, Any]:
   """Gradio-friendly wrapper around run_rag_pipeline.

   Expected UI mapping:
      inputs  -> question, course, optional image, generation params
      outputs -> answer, citations_text, mode, error

   Returns:
      A dictionary with answer text, formatted citations, the selected mode,
      and an error message when something fails.
   """
   # TODO: Connect image_input to the multimodal branch once it exists.
   try:
      result = run_rag_pipeline(
         question=question,
         course=course,
         image_input=image_input,
         top_k=top_k,
         max_new_tokens=max_new_tokens,
         temperature=temperature,
      )
      return {
         "answer": result.get("answer", ""),
         "citations_text": format_citations_for_gradio(result),
         "mode": result.get("mode", "text"),
         "error": None,
      }
   except Exception as exc:
      return {
         "answer": "",
         "citations_text": "",
         "mode": "multimodal" if image_input is not None else "text",
         "error": str(exc),
      }


def format_citations_for_gradio(result: Dict[str, Any]) -> str:
   lines: List[str] = []
   for idx, chunk in enumerate(result.get("retrieved_chunks", []), start=1):
      location = _format_chunk_location(chunk)
      score = chunk.get("score", 0.0)
      lines.append(f"[{idx}] {location} | score={score:.4f}")
   return "\n".join(lines) if lines else "No sources found."

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
