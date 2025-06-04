# BNV

BNV is a simple example project that demonstrates how to fine-tune a small
language model for both text and code generation tasks. The `train.py` script
uses Hugging Face Transformers and the `datasets` library to combine the
`wikitext` dataset with a small code corpus and trains a model based on
`distilgpt2`.

## Requirements

- Python 3.12+
- `torch`, `transformers`, and `datasets`

Install the dependencies with pip:

```bash
pip install torch==2.2.1 torchvision torchaudio transformers datasets
```

## Training

Run the training script to produce the BNV model. The training script supports
adjusting epochs, batch size, and generating a sample after training:

```bash
python -m bnv.train --output ./bnv_model --model distilgpt2 \
    --epochs 3 --batch-size 4 --sample-prompt "def hello():"
```

The resulting model will be saved in the `./bnv_model` directory.

## Generating text or code

After training, use the `bnv.generate` module to produce text or code from
your model. Provide the path to the saved model and a prompt:

```bash
python -m bnv.generate --model ./bnv_model --prompt "Write a Python function to add two numbers"
```

The generation speed depends on your hardware, so responses may take longer
than a few milliseconds on typical machines.

## Capabilities

The `bnv.capabilities` module exposes a set of helper functions built on top of
the trained BNV model. These functions provide convenient prompts for common
text and code tasks:

1. `generate_text` – free‑form text generation
2. `generate_code` – create code from a natural language description
3. `summarize_text` – produce a concise summary of longer passages
4. `translate_text` – translate text into another language
5. `explain_code` – explain what a piece of code does
6. `refactor_code` – suggest improvements for existing code
7. `generate_docstring` – write docstrings for Python functions
8. `answer_question` – respond to questions with optional context
9. `autocomplete_code` – complete partially written code
10. `commit_message` – draft a concise commit message from a diff
11. `style_transfer` – rewrite text in a specified style
12. `code_review` – provide a short code review with improvement tips

Import the module and call these helpers after you have trained or downloaded a
model:

```python
from bnv import capabilities

text = capabilities.summarize_text("./bnv_model", "Long text to summarize")
print(text)
```
