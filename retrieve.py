"""Inspect retrieval against a company knowledge base."""
import argparse
from research import retrieve_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir")
    parser.add_argument("question")
    args = parser.parse_args()
    for doc in retrieve_evidence(args.question, args.index_dir):
        print(doc.metadata)
        print(doc.page_content, "\n")


if __name__ == "__main__":
    main()
