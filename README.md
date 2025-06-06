# Multimodal Model

This repository contains a single Python script `multimodal_model.py` that
implements a minimal demonstration of text, image, video and code generation in
one file. It uses Hugging Face Transformers for text/code generation and the
Diffusers library for image generation. Video output is produced by creating a
series of images and encoding them with `imageio`.

The `MultiModalModel` exposes seven simple APIs:

1. `generate_text` – free-form text generation
2. `generate_code` – produce code from a description
3. `generate_image` – create an image from text
4. `generate_video` – create a short video from a prompt
5. `fetch_wikipedia_summary` – retrieve a summary for a topic
6. `fetch_random_quote` – get a quote from the Quotable API
7. `fetch_weather` – current temperature from Open-Meteo

The script is intentionally small and designed for experimentation only. It does
not provide a full-fledged training pipeline or production-ready API.

## Requirements

- Python 3.10+
- `torch`, `transformers`, `diffusers`, `imageio`

Install the dependencies with:

```bash
pip install torch transformers diffusers imageio
```

## Usage

Run the script directly to generate a short text sample, a small image and a
very short video composed of generated frames:

```bash
python multimodal_model.py
```

Feel free to import the `MultiModalModel` class in your own code and experiment
with any of the seven API methods listed above.
