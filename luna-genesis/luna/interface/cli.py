"""Command-line interface loop for Luna Genesis."""

from __future__ import annotations

from luna.utils.logger import get_logger


logger = get_logger("luna.cli")


def run_cli(process_input):
    """Run interactive shell until user exits."""
    print("Luna Genesis ready. Type 'exit' to quit.")
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = process_input(user_text)
        logger.info("Responded to user input")
        print(f"Luna> {response}")
