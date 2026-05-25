# Assignment 3: Multimodal AI Model Explorer

This project contains Python code for a multimodal AI assignment. It includes:

- A Streamlit web application in `app.py`
- Reusable model/provider functions in `multimodal_models.py`
- A Jupyter notebook template in `Assignment3_Multimodal_Notebook.ipynb`
- Provider setup in `.env.example`

## Models Used

| Modality | Provider | Default model | File/function |
|---|---|---|---|
| Text -> Text | Groq | `llama-3.3-70b-versatile` | `generate_text_response` |
| Text -> Image | Hugging Face | `stabilityai/stable-diffusion-xl-base-1.0` | `generate_image` |
| Image -> Text | Google Gemini | `gemini-1.5-flash` | `describe_image` |
| Text -> Audio | Hugging Face | `facebook/mms-tts-eng` | `generate_audio` |
| Audio -> Text | Groq | `whisper-large-v3-turbo` | `transcribe_audio` |
| Text -> Video | Hugging Face | `damo-vilab/text-to-video-ms-1.7b` | `generate_video` |
| Video -> Text | Google Gemini | `gemini-1.5-flash` | `describe_video` |

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install requirements.

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add API keys.

```bash
copy .env.example .env
```

Required keys:

- `GROQ_API_KEY` for text generation and audio transcription
- `GEMINI_API_KEY` for image/video understanding
- `HF_API_TOKEN` for image, audio, and video generation

4. Run the web application.

```bash
streamlit run app.py
```

## Web App Features

- Text input for text responses
- Text prompt for image generation
- Image upload for image description
- Text prompt for speech generation
- Audio upload or recording for transcription
- Video upload for summary/description
- Text prompt for video generation

## Notebook

Open `Assignment3_Multimodal_Notebook.ipynb` in Jupyter or Colab. Run each section after setting the same API keys used by the app.

The notebook mirrors the assignment workflow:

- Select a model
- Provide sample input
- Call the provider
- Display the output clearly

## Notes

- Text-to-video models can be slow and may require a paid or enabled Hugging Face endpoint.
- You can replace any default model in `.env` or from the Streamlit sidebar.
- Keep API keys private and do not submit your `.env` file.
