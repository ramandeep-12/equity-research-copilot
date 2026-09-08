import hashlib

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
    "Upload financial reports for one company and ask grounded research questions."
)

# --------------------------------------------------
# Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --------------------------------------------------
# Upload financial reports
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Financial Reports",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Detect if uploaded selection has changed
# --------------------------------------------------

if uploaded_files:

    current_signature = tuple(
        sorted(
            hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()
            for uploaded_file in uploaded_files
        )
    )

    previous_signature = st.session_state.get(
        "upload_signature"
    )

    if previous_signature != current_signature:

        st.session_state["upload_signature"] = current_signature

        # Clear previous active workspace
        st.session_state.pop("index_dir", None)
        st.session_state.pop("company_name", None)
        st.session_state.pop("reports", None)
        st.session_state["messages"] = []

    # --------------------------------------------------
    # Show selected files
    # --------------------------------------------------

    st.write("### Selected Reports")

    for uploaded_file in uploaded_files:

        st.write(
            f"- {uploaded_file.name}"
        )

    # --------------------------------------------------
    # Process reports
    # --------------------------------------------------

    if st.button("Process Reports"):

        processed_reports = []

        try:

            with st.spinner(
                "Processing and indexing financial reports..."
            ):

                for uploaded_file in uploaded_files:

                    report_info = ingest_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )

                    processed_reports.append(
                        report_info
                    )

            # --------------------------------------------------
            # Remove duplicate reports
            # --------------------------------------------------

            original_count = len(processed_reports)

            unique_reports = {}

            for report in processed_reports:

                report_id = report["report_id"]

                if report_id not in unique_reports:

                    unique_reports[report_id] = report

            processed_reports = list(
                unique_reports.values()
            )

            duplicate_count = (
                original_count - len(processed_reports)
            )

            if duplicate_count > 0:

                st.info(
                    f"{duplicate_count} duplicate report(s) "
                    f"detected and skipped."
                )

            # --------------------------------------------------
            # Verify all reports belong to same company
            # --------------------------------------------------

            companies = {
                report["company_name"]
                for report in processed_reports
            }

            if len(companies) > 1:

                st.error(
                    "The uploaded reports appear to belong to "
                    "different companies. Please upload reports "
                    "for one company at a time."
                )

            else:

                company_name = processed_reports[0][
                    "company_name"
                ]

                # --------------------------------------------------
                # Verify same company index
                # --------------------------------------------------

                index_dirs = {
                    report["index_dir"]
                    for report in processed_reports
                }

                if len(index_dirs) != 1:

                    st.error(
                        "The reports were not stored in the same "
                        "company knowledge base."
                    )

                else:

                    # --------------------------------------------------
                    # Store active workspace
                    # --------------------------------------------------

                    st.session_state["company_name"] = (
                        company_name
                    )

                    st.session_state["index_dir"] = (
                        processed_reports[0]["index_dir"]
                    )

                    st.session_state["reports"] = (
                        processed_reports
                    )

                    # --------------------------------------------------
                    # New vs reused reports
                    # --------------------------------------------------

                    new_reports = [
                        report
                        for report in processed_reports
                        if not report["reused"]
                    ]

                    reused_reports = [
                        report
                        for report in processed_reports
                        if report["reused"]
                    ]

                    st.success(
                        f"{len(processed_reports)} unique report(s) "
                        f"ready for {company_name}."
                    )

                    if new_reports:

                        st.info(
                            f"{len(new_reports)} new report(s) indexed."
                        )

                    if reused_reports:

                        st.info(
                            f"{len(reused_reports)} report(s) "
                            f"loaded from the existing index."
                        )

        except Exception as e:

            st.error(
                f"Unable to process reports: {e}"
            )


# --------------------------------------------------
# Active research workspace
# --------------------------------------------------

if (
    "index_dir" in st.session_state
    and "reports" in st.session_state
):

    st.divider()

    # --------------------------------------------------
    # Company
    # --------------------------------------------------

    st.subheader(
        f"Research: {st.session_state['company_name']}"
    )

    # --------------------------------------------------
    # Available reports
    # --------------------------------------------------

    st.write("### Available Reports")

    for report in st.session_state["reports"]:

        status = (
            "Existing"
            if report["reused"]
            else "New"
        )

        st.write(
            f"**{report['report_type']}** "
            f"• FY {report['fiscal_year']} "
            f"• {report['filename']} "
            f"• {status}"
        )

        # --------------------------------------------------
    # Research mode
    # --------------------------------------------------

    research_mode = st.radio(
        "Research Mode",
        [
            "Ask Question",
            "Compare Reports"
        ],
        horizontal=True
    )

    # ==================================================
    # ASK QUESTION MODE
    # ==================================================

    if research_mode == "Ask Question":

        # ----------------------------------------------
        # Display chat history
        # ----------------------------------------------

        for message in st.session_state["messages"]:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if message["role"] == "assistant":

                    for source in message.get(
                        "sources",
                        []
                    ):

                        with st.expander(
                            f"{source['report_type']} "
                            f"• FY {source['fiscal_year']} "
                            f"• Page {source['page']}"
                        ):

                            st.write(
                                source["content"]
                            )

        # ----------------------------------------------
        # Chat input
        # ----------------------------------------------

        question = st.chat_input(
            "Ask a research question..."
        )

        if question:

            # Save previous conversation
            chat_history = (
                st.session_state["messages"][-6:]
            )

            # Save/display user question
            st.session_state["messages"].append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(
                    question
                )

            try:

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Analyzing financial reports..."
                    ):

                        result = ask_equity_question(
                            question,
                            st.session_state["index_dir"],
                            chat_history=chat_history
                        )

                    st.markdown(
                        result["answer"]
                    )

                    # ----------------------------------
                    # Sources
                    # ----------------------------------

                    for source in result["sources"]:

                        with st.expander(
                            f"{source['report_type']} "
                            f"• FY {source['fiscal_year']} "
                            f"• Page {source['page']}"
                        ):

                            st.write(
                                source["content"]
                            )

                # Save assistant response
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    }
                )

            except Exception as e:

                st.error(
                    f"Unable to analyze reports: {e}"
                )

    # ==================================================
    # COMPARE REPORTS MODE
    # ==================================================

    elif research_mode == "Compare Reports":

        reports = st.session_state["reports"]

        # Need at least 2 reports
        if len(reports) < 2:

            st.warning(
                "At least two different reports are "
                "required for comparison."
            )

        else:

            st.write(
                "### Compare Financial Reports"
            )

            # ------------------------------------------
            # Build report options
            # ------------------------------------------

            report_options = {}

            for report in reports:

                label = (
                    f"{report['report_type']} "
                    f"• FY {report['fiscal_year']} "
                    f"• {report['filename']}"
                )

                report_options[label] = report

            labels = list(
                report_options.keys()
            )

            # ------------------------------------------
            # Report selectors
            # ------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                report_a_label = st.selectbox(
                    "Report A",
                    labels,
                    index=0
                )

            with col2:

                report_b_label = st.selectbox(
                    "Report B",
                    labels,
                    index=1
                )

            report_a = report_options[
                report_a_label
            ]

            report_b = report_options[
                report_b_label
            ]

            # ------------------------------------------
            # Comparison topic
            # ------------------------------------------

            comparison_topic = st.text_input(
                "What would you like to compare?",
                placeholder=(
                    "e.g. Cloud revenue growth, "
                    "operating income, margins, risks"
                )
            )

            # ------------------------------------------
            # Compare button
            # ------------------------------------------

            if st.button(
                "Compare Reports"
            ):

                if (
                    report_a["report_id"]
                    == report_b["report_id"]
                ):

                    st.warning(
                        "Please select two different reports."
                    )

                elif not comparison_topic.strip():

                    st.warning(
                        "Please enter a comparison topic."
                    )

                else:

                    # ------------------------------------------
                    # Build comparison question
                    # ------------------------------------------

    comparison_question = f"""
Compare {comparison_topic} between these two financial reports:

Report A:
{report_a['report_type']} • FY {report_a['fiscal_year']}

Report B:
{report_b['report_type']} • FY {report_b['fiscal_year']}

Explain:
- the value or disclosure in Report A
- the value or disclosure in Report B
- what changed between the two reports
- the direction and size of the change when the evidence supports it

Use only evidence from these two selected reports.
"""

    try:

        with st.spinner(
            "Comparing selected financial reports..."
        ):

            result = ask_equity_question(
                comparison_question,
                st.session_state["index_dir"],

                report_ids=[
                    report_a["report_id"],
                    report_b["report_id"]
                ]
            )

        # --------------------------------------
        # Comparison result
        # --------------------------------------

        st.subheader(
            "Comparison Analysis"
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
                f"{source['report_type']} "
                f"• FY {source['fiscal_year']} "
                f"• Page {source['page']}"
            ):

                st.write(
                    source["content"]
                )

    except Exception as e:

        st.error(
            f"Unable to compare reports: {e}"
        )
