import streamlit as st

from research import ask_equity_question


st.set_page_config(
    page_title="Equity Research Copilot",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Equity Research Copilot")

st.caption(
    "Ask questions about Microsoft's FY2024 Annual Report"
)


question = st.text_input(
    "Ask a research question",
    placeholder="e.g. What financial risks does Microsoft face?"
)


if st.button("Analyze"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing annual report..."):

            result = ask_equity_question(question)


        st.subheader("Research Analysis")

        st.markdown(result["answer"])


        st.subheader("Sources")

        for source in result["sources"]:

            with st.expander(
                f"Annual Report — Page {source['page']}"
            ):

                st.write(source["content"])