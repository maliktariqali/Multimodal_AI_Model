from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from multimodal_models import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_HF_IMAGE_MODEL,
    DEFAULT_HF_TTS_MODEL,
    DEFAULT_HF_VIDEO_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    MissingCredentialError,
    describe_image,
    describe_video,
    generate_audio,
    generate_image,
    generate_text_response,
    generate_video,
    transcribe_audio,
)


st.set_page_config(
    page_title="Assignment 3 Multimodal AI",
    layout="wide",
)


def show_error(error: Exception) -> None:
    if isinstance(error, MissingCredentialError):
        st.error(str(error))
    else:
        st.error(f"{type(error).__name__}: {error}")


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def main() -> None:
    st.title("Multimodal AI Model Explorer")

    with st.sidebar:
        st.header("Models")
        text_model = st.text_input("Text -> Text", DEFAULT_TEXT_MODEL)
        transcription_model = st.text_input("Audio -> Text", DEFAULT_TRANSCRIPTION_MODEL)
        gemini_model = st.text_input("Image/Video -> Text", DEFAULT_GEMINI_MODEL)
        image_model = st.text_input("Text -> Image", DEFAULT_HF_IMAGE_MODEL)
        audio_model = st.text_input("Text -> Audio", DEFAULT_HF_TTS_MODEL)
        video_model = st.text_input("Text -> Video", DEFAULT_HF_VIDEO_MODEL)

        st.header("Keys")
        st.caption("Configure GROQ_API_KEY, GEMINI_API_KEY, and HF_API_TOKEN in .env.")

    tabs = st.tabs(
        [
            "Text Chat",
            "Image Generator",
            "Image Analyzer",
            "Audio Tools",
            "Video Tools",
        ]
    )

    with tabs[0]:
        st.subheader("Text -> Text")
        prompt = st.text_area(
            "Prompt",
            value="Explain multimodal AI in simple terms with one example.",
            height=160,
        )
        system_prompt = st.text_input(
            "System prompt",
            "You are a concise AI tutor for undergraduate students.",
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

        if st.button("Generate response", type="primary", key="text_response"):
            with st.spinner("Generating text response..."):
                try:
                    response = generate_text_response(
                        prompt,
                        system_prompt=system_prompt,
                        model=text_model,
                        temperature=temperature,
                    )
                    st.markdown(response)
                except Exception as error:
                    show_error(error)

    with tabs[1]:
        st.subheader("Text -> Image")
        image_prompt = st.text_area(
            "Image prompt",
            value="A clean futuristic classroom where students test multimodal AI tools, digital art",
            height=120,
        )
        col_steps, col_guidance = st.columns(2)
        with col_steps:
            steps = st.slider("Inference steps", 10, 60, 30, 1)
        with col_guidance:
            guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, 0.5)

        if st.button("Generate image", type="primary", key="image_generate"):
            with st.spinner("Generating image..."):
                try:
                    result = generate_image(
                        image_prompt,
                        model=image_model,
                        steps=steps,
                        guidance_scale=guidance,
                    )
                    st.image(result.data, caption="Generated image")
                    st.download_button(
                        "Download image",
                        data=result.data,
                        file_name=result.filename,
                        mime=result.mime_type,
                    )
                except Exception as error:
                    show_error(error)

    with tabs[2]:
        st.subheader("Image -> Text")
        image_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_upload",
        )
        image_question = st.text_area(
            "Question for the image",
            value="Describe this image and list the main objects you can identify.",
            height=100,
        )
        if image_file:
            st.image(Image.open(image_file), caption=image_file.name)
            if st.button("Analyze image", type="primary", key="image_analyze"):
                with st.spinner("Analyzing image..."):
                    try:
                        output = describe_image(
                            image_file.getvalue(),
                            image_file.type or "image/png",
                            prompt=image_question,
                            model=gemini_model,
                        )
                        st.markdown(output)
                    except Exception as error:
                        show_error(error)

    with tabs[3]:
        st.subheader("Audio")
        speech_text = st.text_area(
            "Text for speech",
            value="Hello. This audio was generated by a multimodal AI web application.",
            height=100,
        )
        if st.button("Generate audio", type="primary", key="audio_generate"):
            with st.spinner("Generating audio..."):
                try:
                    result = generate_audio(speech_text, model=audio_model)
                    st.audio(result.data, format=result.mime_type)
                    st.download_button(
                        "Download audio",
                        data=result.data,
                        file_name=result.filename,
                        mime=result.mime_type,
                    )
                except Exception as error:
                    show_error(error)

        st.divider()
        st.subheader("Audio -> Text")
        uploaded_audio = st.file_uploader(
            "Upload audio",
            type=["mp3", "wav", "m4a", "ogg", "webm"],
            key="audio_upload",
        )
        audio_input = getattr(st, "audio_input", None)
        recorded_audio = audio_input("Record audio") if audio_input else None
        audio_source = recorded_audio or uploaded_audio

        if audio_source:
            st.audio(audio_source)
            if st.button("Transcribe audio", type="primary", key="audio_transcribe"):
                temp_path = save_uploaded_file(audio_source)
                with st.spinner("Transcribing audio..."):
                    try:
                        transcript = transcribe_audio(
                            temp_path,
                            model=transcription_model,
                        )
                        st.markdown(transcript)
                    except Exception as error:
                        show_error(error)

    with tabs[4]:
        st.subheader("Video -> Text")
        video_file = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "mpeg", "mpg", "webm"],
            key="video_upload",
        )
        video_prompt = st.text_area(
            "Question for the video",
            value="Summarize this video in 5 bullet points and mention visible actions.",
            height=100,
        )
        if video_file:
            st.video(video_file)
            if st.button("Analyze video", type="primary", key="video_analyze"):
                with st.spinner("Analyzing video..."):
                    try:
                        output = describe_video(
                            video_file.getvalue(),
                            video_file.type or "video/mp4",
                            prompt=video_prompt,
                            model=gemini_model,
                        )
                        st.markdown(output)
                    except Exception as error:
                        show_error(error)

        st.divider()
        st.subheader("Text -> Video")
        video_text_prompt = st.text_area(
            "Video prompt",
            value="A short cinematic shot of a robot assistant arranging books in a modern library",
            height=100,
        )
        if st.button("Generate video", type="primary", key="video_generate"):
            with st.spinner("Generating video. This can take several minutes..."):
                try:
                    result = generate_video(video_text_prompt, model=video_model)
                    st.video(result.data)
                    st.download_button(
                        "Download video",
                        data=result.data,
                        file_name=result.filename,
                        mime=result.mime_type,
                    )
                except Exception as error:
                    show_error(error)


if __name__ == "__main__":
    main()
