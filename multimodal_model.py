from __future__ import annotations

"""Simple multimodal generation demo in a single file.

This script demonstrates how one might combine text, image, video and code
generation using Hugging Face libraries. It is intentionally minimal and is not
intended for production use.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import imageio.v2 as imageio
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline


@dataclass
class MultiModalModel:
    """Load models for text and image generation."""

    text_model: AutoModelForCausalLM
    text_tokenizer: AutoTokenizer
    image_pipe: StableDiffusionPipeline

    @classmethod
    def load(
        cls,
        text_model_name: str = "distilgpt2",
        image_model_name: str = "stabilityai/stable-diffusion-2-1",
    ) -> "MultiModalModel":
        """Load the required models."""
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_model = AutoModelForCausalLM.from_pretrained(text_model_name)
        image_pipe = StableDiffusionPipeline.from_pretrained(image_model_name)
        image_pipe = image_pipe.to("cpu")
        return cls(text_model, text_tokenizer, image_pipe)

    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate free-form text."""
        inputs = self.text_tokenizer(prompt, return_tensors="pt")
        outputs = self.text_model.generate(**inputs, max_length=max_length)
        return self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_code(self, description: str, max_length: int = 200) -> str:
        """Generate code from a natural language description."""
        prompt = f"Write Python code for: {description}\nCode:"
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


def main() -> None:
    model = MultiModalModel.load()
    print(model.generate_text("Hello world"))
    print(model.generate_code("a function that adds two numbers"))
    print("Image saved to", model.generate_image("A scenic landscape"))
    print("Video saved to", model.generate_video("A moving square"))


if __name__ == "__main__":
    main()
