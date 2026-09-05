evaluation_questions = [
    "What financial risks does Microsoft face?",
    "How much did Microsoft Cloud revenue grow?",
    "What legal liabilities does Microsoft report?",
    "What currencies expose Microsoft to foreign exchange risk?",
    "What is Microsoft's interest rate risk?",
    "How much did total revenue increase in fiscal 2024?",
    "What drove Intelligent Cloud revenue growth?",
    "How did Office 365 Commercial perform?",
    "What are Microsoft's reportable segments?",
    "What does Microsoft say about artificial intelligence?",
    "How much was Microsoft's operating income?",
    "What are Microsoft's major sources of revenue?"
]


from research import ask_equity_question

for i, question in enumerate(evaluation_questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {i}: {question}")
    print("=" * 80)

    result = ask_equity_question(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:
        print(
            f"- Page {source['page']} "
            f"({source['source']})"
        )