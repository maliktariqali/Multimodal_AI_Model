from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


DEFAULT_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MULTIMODAL_MODEL", "gemini-1.5-flash")
DEFAULT_HF_IMAGE_MODEL = os.getenv("HF_TEXT_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
DEFAULT_HF_TTS_MODEL = os.getenv("HF_TEXT_AUDIO_MODEL", "facebook/mms-tts-eng")
DEFAULT_HF_VIDEO_MODEL = os.getenv("HF_TEXT_VIDEO_MODEL", "damo-vilab/text-to-video-ms-1.7b")


class MissingCredentialError(RuntimeError):
    """Raised when a selected provider needs an API key that is not configured."""


@dataclass(frozen=True)
class BinaryResult:
    data: bytes
    mime_type: str
    filename: str


def _required_env(name: str, purpose: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise MissingCredentialError(
        f"Missing {name}. Add it to your .env file to use {purpose}."
    )


def _import_groq() -> Any:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Install the groq package: pip install groq") from exc
    return Groq


def _import_gemini() -> Any:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "Install the Google Generative AI package: pip install google-generativeai"
        ) from exc
    return genai


def _filename_for(prefix: str, mime_type: str, fallback_ext: str) -> str:
    extension = mimetypes.guess_extension(mime_type.split(";")[0]) or fallback_ext
    if extension == ".jpe":
        extension = ".jpg"
    return f"{prefix}{extension}"


def generate_text_response(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful AI assistant. Keep answers clear and useful.",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    """Text -> Text using Groq chat completions."""
    api_key = _required_env("GROQ_API_KEY", "Groq text generation")
    Groq = _import_groq()
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=model or DEFAULT_TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def transcribe_audio(
    audio_path: str | Path,
    *,
    model: str | None = None,
    language: str | None = None,
) -> str:
    """Audio -> Text using Groq Whisper-compatible transcription."""
    api_key = _required_env("GROQ_API_KEY", "Groq audio transcription")
    Groq = _import_groq()
    client = Groq(api_key=api_key)

    path = Path(audio_path)
    kwargs: dict[str, Any] = {
        "file": (path.name, path.read_bytes()),
        "model": model or DEFAULT_TRANSCRIPTION_MODEL,
        "response_format": "json",
    }
    if language:
        kwargs["language"] = language

    transcription = client.audio.transcriptions.create(**kwargs)
    return getattr(transcription, "text", str(transcription))


def _gemini_generate_with_blob(
    media_bytes: bytes,
    mime_type: str,
    prompt: str,
    *,
    model: str | None = None,
) -> str:
    api_key = _required_env("GEMINI_API_KEY", "Gemini multimodal analysis")
    genai = _import_gemini()
    genai.configure(api_key=api_key)

    gemini_model = genai.GenerativeModel(model or DEFAULT_GEMINI_MODEL)
    response = gemini_model.generate_content(
        [
            prompt,
            {
                "mime_type": mime_type,
                "data": media_bytes,
            },
        ]
    )
    return getattr(response, "text", "") or "No text response returned."


def describe_image(
    image_bytes: bytes,
    mime_type: str,
    *,
    prompt: str = "Describe this image in detail and identify the important objects.",
    model: str | None = None,
) -> str:
    """Image -> Text using Gemini."""
    return _gemini_generate_with_blob(image_bytes, mime_type, prompt, model=model)


def describe_video(
    video_bytes: bytes,
    mime_type: str,
    *,
    prompt: str = "Summarize this video. Mention the scene, actions, and any visible text.",
    model: str | None = None,
) -> str:
    """Video -> Text using Gemini."""
    return _gemini_generate_with_blob(video_bytes, mime_type, prompt, model=model)


def _hugging_face_binary_request(
    model: str,
    payload: dict[str, Any],
    *,
    purpose: str,
    timeout: int = 240,
) -> tuple[bytes, str]:
    api_key = _required_env("HF_API_TOKEN", f"Hugging Face {purpose}")
    url = f"https://api-inference.huggingface.co/models/{model}"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )

    content_type = response.headers.get("content-type", "application/octet-stream")
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Hugging Face {purpose} failed: {detail}")

    if "application/json" in content_type:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"Hugging Face returned JSON instead of binary output for {purpose}: {detail}"
        )

    return response.content, content_type


def generate_image(
    prompt: str,
    *,
    model: str | None = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
) -> BinaryResult:
    """Text -> Image using the Hugging Face Inference API."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        },
        "options": {"wait_for_model": True},
    }
    data, mime_type = _hugging_face_binary_request(
        model or DEFAULT_HF_IMAGE_MODEL,
        payload,
        purpose="text-to-image generation",
    )
    return BinaryResult(
        data=data,
        mime_type=mime_type,
        filename=_filename_for("generated-image", mime_type, ".png"),
    )


def generate_audio(
    prompt: str,
    *,
    model: str | None = None,
) -> BinaryResult:
    """Text -> Audio using the Hugging Face Inference API."""
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }
    data, mime_type = _hugging_face_binary_request(
        model or DEFAULT_HF_TTS_MODEL,
        payload,
        purpose="text-to-audio generation",
    )
    return BinaryResult(
        data=data,
        mime_type=mime_type,
        filename=_filename_for("generated-audio", mime_type, ".wav"),
    )


def generate_video(
    prompt: str,
    *,
    model: str | None = None,
) -> BinaryResult:
    """Text -> Video using the Hugging Face Inference API."""
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }
    data, mime_type = _hugging_face_binary_request(
        model or DEFAULT_HF_VIDEO_MODEL,
        payload,
        purpose="text-to-video generation",
        timeout=600,
    )
    return BinaryResult(
        data=data,
        mime_type=mime_type,
        filename=_filename_for("generated-video", mime_type, ".mp4"),
    )
