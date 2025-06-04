import argparse
import math
from typing import List

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_text_and_code_datasets(train_ratio: float = 0.9) -> tuple[Dataset, Dataset]:
    """Load text and code datasets, merge them and split into train and eval sets."""
    text_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    code_ds = load_dataset("codeparrot/github-jupyter-text", split="train")
    combined = Dataset.from_dict(
        {
            "text": text_ds["text"] + code_ds["content"],
        }
    )
    split = combined.train_test_split(train_ratio, shuffle=True)
    return split["train"], split["test"]


def tokenize_dataset(tokenizer, dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        remove_columns=["text"],
    )


def train(
    output_dir: str = "./bnv_model",
    model_name: str = "distilgpt2",
    epochs: int = 1,
    batch_size: int = 2,
    train_ratio: float = 0.9,
    sample_prompt: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_ds, eval_ds = load_text_and_code_datasets(train_ratio)
    tokenized_train = tokenize_dataset(tokenizer, train_ds)
    tokenized_eval = tokenize_dataset(tokenizer, eval_ds)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"])
    print("Evaluation metrics", metrics)
    trainer.save_model(output_dir)

    if sample_prompt is not None:
        inputs = tokenizer(sample_prompt, return_tensors="pt")
        generated = trainer.model.generate(**inputs, max_length=50)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the BNV model")
    parser.add_argument(
        "--output", default="./bnv_model", help="Directory to store the model"
    )
    parser.add_argument("--model", default="distilgpt2", help="Base model to fine-tune")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Portion of data used for training",
    )
    parser.add_argument(
        "--sample-prompt",
        help="Optional prompt to generate text after training",
    )
    args = parser.parse_args(argv)
    train(
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        sample_prompt=args.sample_prompt,
    )


if __name__ == "__main__":
    main()
