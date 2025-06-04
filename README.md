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
