import json
import logging
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

try:
   from qwen_vl_utils import process_vision_info
except ImportError:
   process_vision_info = None

logger = logging.getLogger(__name__)

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

def _multimodal_stack_ready(model_name: str) -> bool:
   if "VL" not in model_name.upper():
      return False
   if Qwen2_5_VLForConditionalGeneration is None:
      return False
   if process_vision_info is None:
      return False
   return True

def _coerce_image_for_qwen(image_input: Any) -> Any:
   if image_input is None:
      return None
   from PIL import Image

   if isinstance(image_input, Image.Image):
      return image_input.convert("RGB")
   if isinstance(image_input, (str, Path)):
      path = Path(image_input)
      if path.is_file():
         return str(path.resolve())
      return str(image_input)
   if isinstance(image_input, np.ndarray):
      array = image_input
      if array.dtype != np.uint8:
         if array.max() <= 1.0:
            array = (array * 255).clip(0, 255).astype(np.uint8)
         else:
            array = array.astype(np.uint8)
      if array.ndim == 2:
         return Image.fromarray(array).convert("RGB")
      if array.ndim == 3 and array.shape[2] >= 3:
         return Image.fromarray(array[:, :, :3]).convert("RGB")
   return image_input

def generate_answer_multimodal(
   question: str,
   chunks: List[Dict[str, Any]],
   image_input: Any,
   generator_model: str = DEFAULT_GENERATOR_MODEL,
   max_new_tokens: int = 256,
   temperature: float = 0.2,
   top_p: float = 0.9,
) -> str:
   if not _multimodal_stack_ready(generator_model):
      raise RuntimeError("Multimodal stack is not available for this generator model.")

   coerced = _coerce_image_for_qwen(image_input)
   if coerced is None:
      raise ValueError("image_input is empty or could not be converted to an image.")

   processor, model = _load_generator(generator_model)
   context = build_context(chunks) if chunks else "(No matching course chunks were retrieved.)"

   messages = [
      {
         "role": "system",
         "content": (
            "You are an academic study assistant. Answer using the provided context when it is relevant, "
            "and use the attached image when it helps. If the context does not contain the answer, say so clearly. "
            "Cite relevant sources using bracketed numbers like [1], [2]."
         ),
      },
      {
         "role": "user",
         "content": [
            {"type": "image", "image": coerced},
            {
               "type": "text",
               "text": (
                  f"Question: {question}\n\n"
                  f"Context:\n{context}\n\n"
                  "Write a concise, grounded answer with citations when you use the context."
               ),
            },
         ],
      },
   ]

   prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   image_inputs, video_inputs = process_vision_info(messages)
   inputs = processor(
      text=[prompt_text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
   )
   device = next(model.parameters()).device
   inputs = inputs.to(device)

   tokenizer = processor.tokenizer
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

   input_len = inputs["input_ids"].shape[-1]
   generated_tokens = output_ids[0][input_len:]
   answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
   return answer or "The model did not produce a usable answer."

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

   Uses multimodal generation when ``image_input`` is set and the configured
   generator supports Qwen2.5-VL; otherwise uses the text-only path. Multimodal
   failures fall back to text generation without raising.
   """
   retrieved_chunks = retrieve_relevant_chunks(
      query=question,
      processed_dir=processed_dir,
      course=course,
      embedding_model=embedding_model,
      top_k=top_k,
   )

   use_multimodal = image_input is not None and _multimodal_stack_ready(generator_model)
   mode = "text"
   answer = ""

   if use_multimodal:
      try:
         answer = generate_answer_multimodal(
            question=question,
            chunks=retrieved_chunks,
            image_input=image_input,
            generator_model=generator_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
         )
         mode = "multimodal"
      except Exception as exc:
         logger.warning("Multimodal generation failed; falling back to text: %s", exc)
         answer = generate_answer(
            question=question,
            chunks=retrieved_chunks,
            generator_model=generator_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
         )
         mode = "text"
   else:
      answer = generate_answer(
         question=question,
         chunks=retrieved_chunks,
         generator_model=generator_model,
         max_new_tokens=max_new_tokens,
         temperature=temperature,
      )
      mode = "text"

   return {
      "question": question,
      "answer": answer,
      "retrieved_chunks": retrieved_chunks,
      "embedding_model": embedding_model,
      "generator_model": generator_model,
      "course": course,
      "top_k": top_k,
      "mode": mode,
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
         "mode": "text",
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
