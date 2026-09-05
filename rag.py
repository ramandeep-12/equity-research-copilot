from research import ask_equity_question


while True:

    question = input(
        "\nAsk a question about the annual report "
        "(or type 'exit' to quit): "
    )

    if question.lower().strip() in ["exit", "quit"]:
        print("Exiting Equity Research Copilot.")
        break

    result = ask_equity_question(question)

    print("\n" + "=" * 70)
    print("EQUITY RESEARCH ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:

        print(f"\nSource: {source['source']}")
        print(f"Page: {source['page']}")

        print("\nEvidence:")
        print(source["content"][:500])

        print("-" * 70)