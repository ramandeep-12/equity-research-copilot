import streamlit as st

from ingest import ingest_pdf
from research import ask_equity_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Equity Research Copilot",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Equity Research Copilot")

st.caption(
    "Upload a company annual report and ask grounded research questions."
)


# --------------------------------------------------
# Upload report
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Annual Report",
    type=["pdf"]
)


# --------------------------------------------------
# Detect when user selects another PDF
# --------------------------------------------------

if uploaded_file is not None:

    current_file = uploaded_file.name

    previous_file = st.session_state.get(
        "uploaded_filename"
    )


    # New file selected
    if previous_file != current_file:

        st.session_state["uploaded_filename"] = current_file

        # Remove previous report state
        st.session_state.pop("index_dir", None)
        st.session_state.pop("report_name", None)
        st.session_state.pop("company_name", None)
        st.session_state.pop("fiscal_year", None)
        st.session_state.pop("report_type", None)
        st.session_state.pop("pages", None)
        st.session_state.pop("chunks", None)


    st.write(
        f"Selected report: **{uploaded_file.name}**"
    )


    # --------------------------------------------------
    # Process report
    # --------------------------------------------------

    if st.button("Process Report"):

        try:

            with st.spinner(
                "Processing and indexing annual report..."
            ):

                report_info = ingest_pdf(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )


            # Save report information
            st.session_state["index_dir"] = (
                report_info["index_dir"]
            )

            st.session_state["report_name"] = (
                report_info["filename"]
            )

            st.session_state["company_name"] = (
                report_info["company_name"]
            )

            st.session_state["fiscal_year"] = (
                report_info["fiscal_year"]
            )

            st.session_state["report_type"] = (
                report_info["report_type"]
            )

            st.session_state["pages"] = (
                report_info.get("pages")
            )

            st.session_state["chunks"] = (
                report_info.get("chunks")
            )

            st.session_state["reused"] = (
                report_info["reused"]
            )


            if report_info["reused"]:

                st.success(
                    "Report already indexed. Existing index loaded."
                )

            else:

                st.success(
                    f"Report processed successfully — "
                    f"{report_info['pages']} pages, "
                    f"{report_info['chunks']} chunks."
                )


        except Exception as e:

            st.error(
                f"Unable to process report: {e}"
            )


# --------------------------------------------------
# Display detected report metadata
# --------------------------------------------------

if "index_dir" in st.session_state:

    st.write("### Detected Report")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Company",
            st.session_state["company_name"]
        )


    with col2:

        st.metric(
            "Fiscal Year",
            st.session_state["fiscal_year"]
        )


    with col3:

        st.metric(
            "Report Type",
            st.session_state["report_type"]
        )


    st.caption(
        f"File: {st.session_state['report_name']}"
    )


# --------------------------------------------------
# Research section
# --------------------------------------------------

if "index_dir" in st.session_state:

    st.divider()


    st.subheader(
        f"Research: {st.session_state['company_name']}"
    )


    st.caption(
        f"{st.session_state['report_type']} "
        f"• FY {st.session_state['fiscal_year']} "
        f"• {st.session_state['report_name']}"
    )


    question = st.text_input(
        "Ask a research question",
        placeholder="e.g. What drove revenue growth?"
    )


    # --------------------------------------------------
    # Analyze question
    # --------------------------------------------------

    if st.button("Analyze"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing annual report..."
                ):

                    result = ask_equity_question(
                        question,
                        st.session_state["index_dir"]
                    )


                # --------------------------------------
                # Answer
                # --------------------------------------

                st.subheader(
                    "Research Analysis"
                )

                st.markdown(
                    result["answer"]
                )


                # --------------------------------------
                # Sources
                # --------------------------------------

                st.subheader(
                    "Sources"
                )


                if not result["sources"]:

                    st.info(
                        "No supporting source excerpts were returned."
                    )


                for source in result["sources"]:

                    with st.expander(
                        f"{source['source']} — "
                        f"Page {source['page']}"
                    ):

                        st.write(
                            source["content"]
                        )


            except Exception as e:

                st.error(
                    f"Unable to analyze the report: {e}"
                )