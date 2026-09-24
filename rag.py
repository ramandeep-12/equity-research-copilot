"""Command line research using an existing company index."""

import argparse

from research import ask_equity_question, citation_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", help="Company index folder containing metadata.json")
    args = parser.parse_args()
    history = []
    while True:
        question = input("\nResearch question (exit to quit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = ask_equity_question(question, args.index_dir, history)
        print(result["answer"])
        for source in result["sources"]:
            print(f"[{source['id']}] {citation_label(source)}")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ]
        )


if __name__ == "__main__":
    main()
