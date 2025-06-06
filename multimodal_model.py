from __future__ import annotations

"""Simple multimodal generation demo in a single file.

This script demonstrates how one might combine text, image, video and code
generation using Hugging Face libraries. It is intentionally minimal and is not
intended for production use.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import requests

import imageio.v2 as imageio
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS


@dataclass
class PromptSystem:
    """Simple prompt management for the different tools."""

    templates: Dict[str, str] = field(default_factory=lambda: {
        "generate_code": "Write Python code for: {description}\nCode:",
        "summarize_text": "Summarize the following text:\n{text}\nSummary:",
        "translate_text": "Translate this text to French:\n{text}\nFrench:",
        "answer_question": (
            "Context: {context}\nQuestion: {question}\nAnswer:"
        ),
    })

    def get(self, name: str, **kwargs: str) -> str:
        template = self.templates.get(name, "{text}")
        return template.format(**kwargs)


@dataclass
class MultiModalModel:
    """Load models for text and image generation."""

    text_model: AutoModelForCausalLM
    text_tokenizer: AutoTokenizer
    image_pipe: StableDiffusionPipeline
    summarizer: object
    translator: object
    qa: object
    sentiment: object
    prompt_system: PromptSystem

    @classmethod
    def load(
        cls,
        text_model_name: str = "distilgpt2",
        image_model_name: str = "stabilityai/stable-diffusion-2-1",
        prompt_system: PromptSystem | None = None,
    ) -> "MultiModalModel":
        """Load the required models."""
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_model = AutoModelForCausalLM.from_pretrained(text_model_name)
        image_pipe = StableDiffusionPipeline.from_pretrained(image_model_name)
        image_pipe = image_pipe.to("cpu")
        summarizer = pipeline(
            "summarization", model="sshleifer/distilbart-cnn-12-6"
        )
        translator = pipeline(
            "translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"
        )
        qa = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",
        )
        sentiment = pipeline("sentiment-analysis")
        if prompt_system is None:
            prompt_system = PromptSystem()
        return cls(
            text_model,
            text_tokenizer,
            image_pipe,
            summarizer,
            translator,
            qa,
            sentiment,
            prompt_system,
        )

    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate free-form text."""
        inputs = self.text_tokenizer(prompt, return_tensors="pt")
        outputs = self.text_model.generate(**inputs, max_length=max_length)
        return self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_code(self, description: str, max_length: int = 200) -> str:
        """Generate code from a natural language description."""
        prompt = self.prompt_system.get("generate_code", description=description)
        return self.generate_text(prompt, max_length)

    def generate_image(self, prompt: str, output: str = "image.png") -> str:
        """Generate an image file from a text prompt."""
        image = self.image_pipe(prompt).images[0]
        path = Path(output)
        image.save(path)
        return str(path)

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 4,
        fps: int = 1,
        output: str = "video.mp4",
    ) -> str:
        """Create a simple video by generating several images."""
        frames: List[imageio.Image] = []
        for i in range(num_frames):
            frame = self.image_pipe(f"{prompt}, frame {i}").images[0]
            frames.append(frame)
        path = Path(output)
        imageio.mimsave(path, frames, fps=fps)
        return str(path)

    # --- Live data helper methods ---
    def fetch_wikipedia_summary(self, topic: str) -> str:
        """Return a short summary from Wikipedia."""
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("extract", "")
        except requests.RequestException:
            return ""

    def fetch_random_quote(self) -> str:
        """Return a random quote from the Quotable API."""
        try:
            resp = requests.get("https://api.quotable.io/random", timeout=10)
            resp.raise_for_status()
            return resp.json().get("content", "")
        except requests.RequestException:
            return ""

    def fetch_weather(self, latitude: float, longitude: float) -> str:
        """Retrieve a simple weather report from Open-Meteo."""
        url = (
            "https://api.open-meteo.com/v1/forecast?latitude="
            f"{latitude}&longitude={longitude}&current=temperature_2m"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            temp = data.get("current", {}).get("temperature_2m")
            if temp is None:
                return ""
            return f"Current temperature: {temp}°C"
        except requests.RequestException:
            return ""

    # --- Additional AI tools ---
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize a block of text."""
        prompt = self.prompt_system.get("summarize_text", text=text)
        summary = self.summarizer(prompt, max_length=max_length)
        return summary[0]["summary_text"]

    def translate_text(self, text: str) -> str:
        """Translate English text to French."""
        prompt = self.prompt_system.get("translate_text", text=text)
        result = self.translator(prompt)
        return result[0]["translation_text"]

    def answer_question(self, question: str, context: str) -> str:
        """Answer a question given some context."""
        self.prompt_system.get("answer_question", question=question, context=context)
        result = self.qa(question=question, context=context)
        return result["answer"]

    def analyze_sentiment(self, text: str) -> str:
        """Return sentiment label for the text."""
        result = self.sentiment(text)
        return result[0]["label"]

    def text_to_speech(self, text: str, output: str = "speech.mp3") -> str:
        """Convert text to a spoken MP3 file."""
        try:
            tts = gTTS(text)
            path = Path(output)
            tts.save(str(path))
            return str(path)
        except Exception:
            return ""


def main() -> None:
    model = MultiModalModel.load()
    print(model.generate_text("Hello world"))
    print(model.generate_code("a function that adds two numbers"))
    print("Image saved to", model.generate_image("A scenic landscape"))
    print("Video saved to", model.generate_video("A moving square"))
    print("Quote:", model.fetch_random_quote())
    print("Wiki:", model.fetch_wikipedia_summary("Artificial_intelligence"))
    print("Weather:", model.fetch_weather(35.0, 139.0))
    print(
        "Summary:",
        model.summarize_text(
            "OpenAI develops artificial intelligence to benefit humanity."
        ),
    )
    print("Translation:", model.translate_text("Hello world"))
    print(
        "Answer:",
        model.answer_question(
            "What does OpenAI develop?",
            "OpenAI develops artificial intelligence technologies.",
        ),
    )
    print("Sentiment:", model.analyze_sentiment("I love using this model!"))
    print("Speech saved to", model.text_to_speech("Hello from the model"))


if __name__ == "__main__":
    main()
