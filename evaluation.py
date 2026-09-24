"""Manual live evaluation. Requires an API key and an indexed company."""

import argparse

from research import ask_equity_question

QUESTIONS = [
    "What are the major financial risks?",
    "How has revenue growth changed?",
    "What drove operating income?",
    "How did operating cash flow change?",
]


def main() -> None:
    """Print representative answers and citations for manual review, without scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir")
    args = parser.parse_args()
    for question in QUESTIONS:
        result = ask_equity_question(question, args.index_dir)
        print(f"\n{question}\n{result['answer']}\nSources: {result['used_sources']}")


if __name__ == "__main__":
    main()
