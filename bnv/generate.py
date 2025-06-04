import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(model_path: str, prompt: str, max_length: int = 200) -> None:
    """Generate text or code given a prompt using a trained model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate text or code using a trained BNV model"
    )
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of generated text",
    )
    args = parser.parse_args(argv)
    generate(args.model, args.prompt, args.max_length)


if __name__ == "__main__":
    main()
