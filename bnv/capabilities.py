from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def _generate(model_path: str, prompt: str, max_length: int = 200) -> str:
    """Helper to load the model and generate text."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_text(model_path: str, prompt: str, max_length: int = 200) -> str:
    """Free-form text generation."""
    return _generate(model_path, prompt, max_length)


def generate_code(model_path: str, description: str, max_length: int = 200) -> str:
    """Generate code from a natural language description."""
    prompt = f"Write code for the following task:\n{description}\nCode:"
    return _generate(model_path, prompt, max_length)


def summarize_text(model_path: str, text: str, max_length: int = 150) -> str:
    """Summarize a block of text."""
    prompt = f"Summarize the following text:\n{text}\nSummary:"
    return _generate(model_path, prompt, max_length)


def translate_text(model_path: str, text: str, target_language: str, max_length: int = 200) -> str:
    """Translate text into the target language."""
    prompt = f"Translate the following text to {target_language}:\n{text}\nTranslation:"
    return _generate(model_path, prompt, max_length)


def explain_code(model_path: str, code: str, max_length: int = 200) -> str:
    """Generate a plain-language explanation of the given code."""
    prompt = f"Explain what the following code does:\n{code}\nExplanation:"
    return _generate(model_path, prompt, max_length)


def refactor_code(model_path: str, code: str, max_length: int = 200) -> str:
    """Suggest improvements to the given code."""
    prompt = f"Improve the following code for clarity and performance:\n{code}\nImproved code:"
    return _generate(model_path, prompt, max_length)


def generate_docstring(model_path: str, code: str, max_length: int = 150) -> str:
    """Generate a docstring for the given code snippet."""
    prompt = f"Add a detailed Python docstring to the following function:\n{code}\n\n"""
    return _generate(model_path, prompt, max_length)


def answer_question(model_path: str, question: str, context: str | None = None, max_length: int = 200) -> str:
    """Answer a question, optionally using additional context."""
    if context:
        prompt = f"Use the context to answer the question.\nContext:{context}\nQuestion:{question}\nAnswer:"
    else:
        prompt = f"Question:{question}\nAnswer:"
    return _generate(model_path, prompt, max_length)


def autocomplete_code(model_path: str, prefix: str, max_length: int = 100) -> str:
    """Complete a partial code snippet."""
    prompt = f"Complete the following code:\n{prefix}"
    return _generate(model_path, prompt, max_length)


def commit_message(model_path: str, diff: str, max_length: int = 100) -> str:
    """Generate a commit message describing the diff."""
    prompt = f"Write a concise git commit message for the following changes:\n{diff}\nMessage:"
    return _generate(model_path, prompt, max_length)


def style_transfer(model_path: str, text: str, style: str, max_length: int = 200) -> str:
    """Rewrite text in a specified style."""
    prompt = f"Rewrite the following text in {style} style:\n{text}\nRewritten:"
    return _generate(model_path, prompt, max_length)


def code_review(model_path: str, code: str, max_length: int = 200) -> str:
    """Provide a short code review."""
    prompt = f"Review the following code and suggest improvements:\n{code}\nReview:"
    return _generate(model_path, prompt, max_length)
