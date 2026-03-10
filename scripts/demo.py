"""Interactive demo — type a natural language request, see the function call.

Usage:
    python scripts/demo.py --adapter checkpoints/dpo
    python scripts/demo.py --adapter checkpoints/sft --stream
    python scripts/demo.py --adapter checkpoints/dpo --schema data/samples/function_schemas.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.pipeline import FunctionCallingPipeline

_DEFAULT_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_products",
        "description": "Search the product catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "max_price": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["title", "date"],
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive function-calling demo")
    parser.add_argument("--adapter", default="./checkpoints/dpo")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as generated")
    parser.add_argument("--schema", default=None, help="Path to JSON file with function schemas")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    functions = _DEFAULT_FUNCTIONS
    if args.schema:
        functions = json.loads(Path(args.schema).read_text())

    print(f"\nLoading adapter from {args.adapter} ...")
    pipeline = FunctionCallingPipeline(
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
    )

    fn_names = [f["name"] for f in functions]
    print(f"\nAvailable functions: {', '.join(fn_names)}")
    print("Type your request (or 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        if args.stream:
            print("Assistant: ", end="", flush=True)
            output_tokens = []
            for token in pipeline.stream(user_input, functions):
                print(token, end="", flush=True)
                output_tokens.append(token)
            print()
            raw = "".join(output_tokens)
        else:
            result = pipeline.call(user_input, functions)
            raw = result.raw_output

            if result.valid:
                print(f"\nFunction call: {result.function_name}")
                print(f"Arguments:     {json.dumps(result.arguments, indent=2)}")
                print(f"Latency:       {result.latency_ms:.0f} ms")
            else:
                print(f"\n[Invalid JSON output]\nRaw: {raw}")
        print()


if __name__ == "__main__":
    main()
