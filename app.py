"""Equity Research Copilot — Streamlit entry point."""

import csv
import io
import logging
import os
import re
import uuid
from pathlib import Path

import streamlit as st

from dashboard import (
    SCALES,
    automatic_changes,
    dashboard_signature,
    load_dashboard,
    period_order,
    series_groups,
)
from ingest import ingest_pdf, list_companies, prepare_report
from research import ask_equity_question, citation_label
from settings import data_root
from source_preview import render_pdf_page
from upload_queue import file_key

st.set_page_config(page_title="Equity Research Copilot", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 900px; padding-top: 6rem; padding-bottom: 2rem;}
.stApp {background: #fafbff;}
[data-testid="stHeader"] {display: none;}
.app-header {position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
    min-height: 64px; padding: 18px 24px; box-sizing: border-box;
    background: #ffffff; border-bottom: 1px solid #e8ecf3;
    color: #1f2937; font-size: 18px; font-weight: 600;}

h1 {font-size: clamp(1.65rem, 4vw, 2.3rem) !important; font-weight: 650 !important;}
[data-testid="stFileUploaderDropzone"] {background: transparent; border: none; padding: 0;}
[data-testid="stFileUploaderDropzoneInstructions"] {display: none;}
.st-key-chat-composer {border-radius: 28px; background: white; padding: 14px 18px;
    box-shadow: 0 4px 20px #2437560a; margin-top: 20px;}
.st-key-chat-composer [data-testid="stForm"] {border: none; padding: 0;}
.st-key-chat-composer textarea {border-radius: 18px; background: white;}
.st-key-chat-upload [data-testid="stFileUploaderDropzone"] button:is([data-testid="stBaseButton-secondary"], [aria-label="Add files"]) {
    border-radius: 50%; width: 44px; min-width: 44px; height: 44px; padding: 0;
    background: #edf1f8; border: none; font-size: 0;}
.st-key-chat-upload [data-testid="stFileUploaderDropzone"] button:is([data-testid="stBaseButton-secondary"], [aria-label="Add files"]) * {display: none;}
.st-key-chat-upload [data-testid="stFileUploaderDropzone"] button:is([data-testid="stBaseButton-secondary"], [aria-label="Add files"])::after {
    content: '+'; font-size: 28px; color: #334155;}
.st-key-chat-upload [data-testid="stFileUploaderDropzone"] button:is([data-testid="stBaseButton-secondary"], [aria-label="Add files"]):hover {background: #dfe7f4;}
.st-key-chat-upload [data-testid="stFileUploaderDropzone"] button:is([data-testid="stBaseButton-secondary"], [aria-label="Add files"]):focus-visible {
    outline: 2px solid #2563eb; outline-offset: 3px;}

[data-testid="stForm"] {background: white; border-radius: 20px; padding: 16px;}
[data-testid="stChatMessage"] {border-radius: 16px; background: #f0f3fa;}
[data-testid="stExpander"] {border-radius: 14px; background: white;}
@media (max-width: 640px) {
    .block-container {padding: 5.5rem 1rem 1.25rem;}
    .app-header {font-size: 16px; padding: 18px 16px;}
}
</style>
""",
    unsafe_allow_html=True,
)

# Streamlit Cloud secrets and local .env both work. Never display the key.
try:
    for setting in ("OPENAI_API_KEY", "OPENAI_MODEL", "EQUITY_DATA_DIR", "EQUITY_WORKSPACE_MODE"):
        if setting in st.secrets:
            os.environ[setting] = str(st.secrets[setting])
except FileNotFoundError:
    pass

st.session_state.setdefault("workspace_id", uuid.uuid4().hex)
persistent = os.getenv("EQUITY_WORKSPACE_MODE", "session") == "persistent"
root = data_root() if persistent else data_root() / "sessions" / st.session_state.workspace_id
ready = bool(os.getenv("OPENAI_API_KEY"))


def show_error(action, error):
    logging.getLogger(__name__).error("%s failed (%s)", action, type(error).__name__)
    if isinstance(error, ValueError):
        st.error(str(error))
    else:
        st.error(
            f"{action} could not finish. Please try again. "
            "Your previously indexed reports are still available."
        )


def report_label(report):
    return (
        f"FY{report['fiscal_year']} · {report.get('fiscal_period', 'Unknown')} · "
        f"{report['report_type']} · {report['filename']}"
    )


def render_sources(sources, key):
    if not sources:
        return
    with st.expander("View answer sources"):
        index = st.selectbox(
            "Cited page",
            range(len(sources)),
            format_func=lambda i: (
                f"[{sources[i].get('id', 'Source')}] PDF page {sources[i]['page']}"
            ),
            key=f"{key}-page",
        )
        source = sources[index]
        st.caption(citation_label(source))
        pdf = Path(active["index_dir"]) / f"{source['report_id']}.pdf"
        preview = None
        if pdf.exists():
            try:
                preview = render_pdf_page(pdf, source["page"])
            except (ValueError, RuntimeError, OSError):
                st.caption(
                    "The page preview is unavailable. You can read the extracted text below."
                )
        else:
            st.caption("The original PDF is unavailable for this saved report.")
        if preview:
            st.image(preview, width="stretch")
            st.caption("Original PDF page — table columns, headings and units are preserved.")
            st.download_button(
                "Download page image",
                preview,
                file_name=f"source-page-{source['page']}.png",
                mime="image/png",
                key=f"{key}-image",
            )
        if st.checkbox("Show retrieved text", key=f"{key}-text", value=not bool(preview)):
            st.caption(
                "These are the excerpts used for the answer. PDF extraction may split table columns across lines."
            )
            with st.container(height=240):
                st.text(source["content"])


def readable_markdown(text):
    """Keep currency dollar signs literal instead of opening LaTeX math spans."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def export_answer(result):
    sources = "\n\n".join(
        f"[{s.get('id', 'Source')}] {citation_label(s)}\n{s['content']}"
        for s in result.get("sources", [])
    )
    return readable_markdown(result["answer"]) + "\n\n## Sources\n\n" + readable_markdown(sources)


try:
    companies = list_companies(root)
except Exception as exc:
    show_error("Loading the report library", exc)
    companies = []

saved_reports = {
    c["index_dir"] + "::" + r["report_id"]: (c, r) for c in companies for r in c["reports"]
}
if st.session_state.get("activate_report"):
    st.session_state["report_picker"] = st.session_state.pop("activate_report")

st.markdown(
    '<header class="app-header" aria-label="Application">Equity Research Copilot</header>',
    unsafe_allow_html=True,
)

active = None
report_key = st.session_state.get("report_picker")
if report_key not in saved_reports:
    report_key = next(iter(saved_reports), None)
if report_key is not None:
    company, report = saved_reports[report_key]
    active = {**company, "reports": [report]}

if not ready:
    st.info("Report analysis is currently unavailable. Please try again later.")

reports = active["reports"] if active else []

content_area = st.container()
composer = st.container(key="chat-composer", border=True)
with composer:
    message_control = st.container()
    file = None
    if not active:
        file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="chat-upload",
        )

st.session_state.setdefault("upload_attempts", {})
signature = file_key(file.getvalue()) if file is not None else None
attempt = st.session_state.upload_attempts.get(signature)
if file is not None and attempt and attempt.get("error"):
    st.error(attempt["error"])
    if active:
        st.caption("Upload not accepted. Your current PDF is still open.")
    if st.button("Retry upload", disabled=not ready):
        st.session_state.upload_attempts.pop(signature, None)
        st.rerun()
if file is not None and ready and signature not in st.session_state.upload_attempts:
    st.session_state.upload_attempts[signature] = {"error": None}
    try:
        with st.spinner("Reading your PDF…"):
            detail = prepare_report(file.getvalue(), file.name, root)
            result = ingest_pdf(file.getvalue(), detail["filename"], detail, root)
            st.session_state["activate_report"] = result["index_dir"] + "::" + detail["report_id"]
        st.rerun()
    except Exception as exc:
        st.session_state.upload_attempts[signature] = {
            "error": str(exc) if isinstance(exc, ValueError) else "Upload failed. Please try again."
        }
        st.rerun()

with content_area:
    if not active:
        st.subheader("What would you like to understand?")
        st.write("Upload an annual or quarterly report to start a conversation.")
        st.caption("Ask about revenue, key risks, or changes from last year.")
    else:
        report = reports[0]
        st.caption(
            f"Chatting with {report['filename']} · {active['company_name']} · FY{report['fiscal_year']}"
        )
        chat_tab = st.container()
        summary_tab = st.expander("Report summary", expanded=False)
        sources_tab = st.expander("Report sources and downloads", expanded=False)
        with sources_tab:
            pdf = Path(active["index_dir"]) / f"{report['report_id']}.pdf"
            if pdf.exists():
                st.download_button(
                    "Download original PDF",
                    pdf.read_bytes(),
                    report["filename"],
                    "application/pdf",
                    key=f"report-{report['report_id']}",
                )

        dashboard_key = "dashboard-" + active["index_dir"] + dashboard_signature(reports)
        if ready and dashboard_key not in st.session_state:
            try:
                with st.status(
                    "Analyzing financial performance, segments and risks…", expanded=True
                ) as status:

                    def progress(index, total, name):
                        status.update(label=f"Researching report {index + 1} of {total}: {name}")

                    st.session_state[dashboard_key] = load_dashboard(
                        active["index_dir"], progress=progress, report_ids=[reports[0]["report_id"]]
                    )
                    status.update(label="Research dashboard ready", state="complete")
            except Exception:
                st.session_state[dashboard_key] = {
                    "errors": [
                        {
                            "report": "Dashboard",
                            "message": "Analysis could not finish. Please try again.",
                        }
                    ],
                    "analyzed": 0,
                    "total": len(reports),
                    **{
                        key: []
                        for key in [
                            "insights",
                            "metrics",
                            "segments",
                            "risks",
                            "drivers",
                            "sources",
                        ]
                    },
                }
        data = st.session_state.get(dashboard_key)
        if data:
            summary_tab.caption(f"{data['analyzed']} of {data['total']} reports analyzed")
            for error in data["errors"]:
                st.warning(f"{error['report']}: {error['message']}")
            if data["errors"] and st.button("Retry incomplete analysis", disabled=not ready):
                st.session_state.pop(dashboard_key, None)
                st.rerun()
            sources = {source["id"]: source for source in data["sources"]}

            def finding_sources(ids, key):
                pages = sorted({sources[sid]["page"] for sid in ids if sid in sources})
                if pages:
                    st.caption("Source: PDF page " + ", ".join(map(str, pages)))

            def render_findings(items, prefix, limit=3):
                if not items:
                    st.caption("Not enough verified evidence in the analyzed reports.")
                for i, finding in enumerate(items[:limit]):
                    st.markdown(readable_markdown("• " + finding["text"]))
                    finding_sources(
                        finding.get("source_ids", [finding.get("source_id")]), f"{prefix}-{i}"
                    )

            with summary_tab:
                groups = series_groups(data["metrics"])
                changes = automatic_changes(groups)
                st.subheader("Key insights")
                # Computed changes are shown first; report findings provide the disclosed context.
                render_findings(changes[:3] + data["insights"], "insight")

                st.subheader("Key metrics")
                available_periods = sorted(
                    {(r["fiscal_year"], r["period"]) for r in data["metrics"]},
                    key=lambda pair: period_order({"fiscal_year": pair[0], "period": pair[1]}),
                    reverse=True,
                )
                if available_periods:
                    period = st.selectbox(
                        "Reporting period",
                        available_periods,
                        format_func=lambda value: f"FY{value[0]} · {value[1]}",
                        key=f"period-{dashboard_key}",
                    )
                    period_rows = [
                        r for r in data["metrics"] if (r["fiscal_year"], r["period"]) == period
                    ]
                    main_metrics = ["Revenue", "Net income", "Diluted EPS"]
                    for name in main_metrics:
                        candidates = [
                            r
                            for r in period_rows
                            if r["name"] == name and r["scope"].casefold() == "consolidated"
                        ]
                        identities = {
                            (
                                r["value"] * SCALES[r["unit"]],
                                r["currency"],
                                r["basis"],
                                r["duration_months"],
                                r["unit"] if r["unit"] in {"per share", "percent"} else "amount",
                            )
                            for r in candidates
                        }
                        with st.container(border=True):
                            if not candidates:
                                st.write(f"**{name}:** Not found in the verified excerpts")
                            elif len(identities) != 1:
                                st.write(
                                    f"**{name}:** Multiple reported values — see the financial table below."
                                )
                            else:
                                row = candidates[0]
                                unit = "per share" if row["unit"] == "per share" else row["unit"]
                                currency = "" if unit == "percent" else row["currency"] + " "
                                st.write(f"**{name}: {currency}{row['value']:,.2f} {unit}**")
                                st.caption(
                                    f"{row['duration_months']} months · {row['basis']} · PDF page {sources[row['source_id']]['page']}"
                                )

                else:
                    st.info(
                        "No financial figures could be verified. You can still research the disclosures in the copilot."
                    )

                st.subheader("Financial trends")
                chart_groups = [group for group in groups if len(group["points"]) >= 2]
                if not chart_groups:
                    st.info(
                        "Trends need at least two verified, comparable periods. Comparative figures within one report can also provide a trend."
                    )
                chart_columns = st.columns(2)
                for i, group in enumerate(chart_groups[:2]):
                    with chart_columns[i % 2]:
                        name, scope, period, duration, currency, basis, kind = group["key"]
                        st.markdown(f"**{name}** · {scope}")
                        divisor = 1e6 if kind == "amount" else 1
                        unit = (
                            f"{currency} millions"
                            if kind == "amount"
                            else "%"
                            if kind == "percent"
                            else f"{currency} per share"
                        )
                        st.caption(f"{unit} · {duration}-month {period} periods · {basis}")
                        chart_data = [
                            {
                                "Period": point["period_label"],
                                "Value": point["normalized_value"] / divisor,
                            }
                            for point in group["points"]
                        ]
                        st.line_chart(chart_data, x="Period", y="Value", color="#2563eb")
                        with st.expander("Trend data and citations"):
                            st.dataframe(chart_data, hide_index=True, width="stretch")
                            finding_sources(
                                [point["source_id"] for point in group["points"]], f"trend-{i}"
                            )
                for group in groups:
                    if group["conflicts"]:
                        st.caption(
                            f"{group['key'][0]}: conflicting reported values for "
                            + ", ".join(map(str, group["conflicts"]))
                            + " were excluded from trend calculations. Review the financial table for restatements or rounding differences."
                        )

                st.subheader("Key risks")
                render_findings(data["risks"], "risk")
                with st.expander("Business segments and growth"):
                    st.subheader("Segment performance")
                    render_findings(data["segments"], "segment")
                    st.subheader("Growth drivers")
                    render_findings(data["drivers"], "driver")

            with sources_tab:
                st.subheader("Sources and evidence")
                pages = {}
                for source in data["sources"]:
                    pages.setdefault((source["report_id"], source["page"]), []).append(source)
                if not pages:
                    st.caption("No verified source excerpts are available yet.")
                else:
                    page_key = st.selectbox(
                        "Source page",
                        sorted(pages),
                        format_func=lambda value: f"PDF page {value[1]}",
                        key=f"source-page-{dashboard_key}",
                    )
                    page_sources = pages[page_key]
                    page_ids = {source["id"] for source in page_sources}
                    st.caption(
                        "Page numbers include the PDF cover. Figures below retain the reported units."
                    )
                    page_metrics = [row for row in data["metrics"] if row["source_id"] in page_ids]
                    if page_metrics:
                        st.markdown("**Figures from this page**")
                        table = [
                            {
                                "Metric": row["name"],
                                "Period": f"FY{row['fiscal_year']} · {row['period']}",
                                "Value": f"{row['value']:,.2f}",
                                "Units": f"{row['currency']} {row['unit']}",
                                "Scope": row["scope"],
                                "Basis": row["basis"],
                                "Months": row["duration_months"],
                            }
                            for row in sorted(
                                page_metrics, key=lambda row: (row["name"], -row["fiscal_year"])
                            )
                        ]
                        st.dataframe(table, hide_index=True, width="stretch")
                    findings = [
                        item
                        for section in ["insights", "segments", "risks", "drivers"]
                        for item in data[section]
                        if item.get("source_id") in page_ids
                    ]
                    quotes = list(
                        dict.fromkeys(item["quote"] for item in findings if item.get("quote"))
                    )
                    if quotes:
                        st.markdown("**Supporting quotes**")
                        for quote in quotes:
                            st.markdown("> " + readable_markdown(" ".join(quote.split())))
                    if not page_metrics and not quotes:
                        st.caption("Open the original excerpt below to read this source.")
                    if st.checkbox(
                        "Show original extracted text", key=f"raw-source-{dashboard_key}"
                    ):
                        st.caption(
                            "PDF extraction can split table columns across lines. This is the original retrieved text."
                        )
                        with st.container(height=300):
                            st.text(
                                "\n\n[…]\n\n".join(
                                    dict.fromkeys(source["content"] for source in page_sources)
                                )
                            )

            with sources_tab:
                if data["metrics"]:
                    with st.expander("Financial data & export"):
                        columns = [
                            "name",
                            "scope",
                            "fiscal_year",
                            "period",
                            "duration_months",
                            "value",
                            "unit",
                            "currency",
                            "basis",
                            "source_id",
                        ]
                        st.dataframe(
                            [{key: row[key] for key in columns} for row in data["metrics"]],
                            hide_index=True,
                            width="stretch",
                        )
                        st.caption(
                            "Reported values retain their original scale. Charts normalize monetary scales and separate reporting bases. Verify source excerpts before relying on classifications."
                        )
                        output = io.StringIO()
                        fields = list(data["metrics"][0]) + ["source_report", "pdf_page"]
                        writer = csv.DictWriter(output, fieldnames=fields)
                        writer.writeheader()
                        for row in data["metrics"]:
                            source = sources[row["source_id"]]
                            exported = {
                                **row,
                                "source_report": source.get("source"),
                                "pdf_page": source["page"],
                            }
                            writer.writerow(
                                {
                                    key: "'" + value
                                    if isinstance(value, str)
                                    and value.startswith(("=", "+", "-", "@"))
                                    else value
                                    for key, value in exported.items()
                                }
                            )
                        st.download_button(
                            "Download financial metrics CSV",
                            output.getvalue(),
                            "financial-metrics.csv",
                            "text/csv",
                        )

        if not data:
            summary_tab.info("The report summary is not available yet.")
        with chat_tab:
            selected = [reports[0]["report_id"]]
            chat_key = "chat-" + active["index_dir"] + "-" + "-".join(sorted(selected))
            messages = st.session_state.setdefault(chat_key, [])
            if messages:
                c1, c2 = st.columns(2)
                if c1.button("Clear conversation"):
                    st.session_state[chat_key] = []
                    st.rerun()
                c2.download_button(
                    "Export research notes",
                    "\n\n".join(
                        f"## {m['role'].title()}\n\n"
                        + export_answer({"answer": m["content"], "sources": m.get("sources", [])})
                        for m in messages
                    ),
                    "research-notes.md",
                    "text/markdown",
                )
            else:
                st.subheader("What would you like to know?")
                st.caption("Try: What were the main risks? Or: How did revenue change?")
            for i, message in enumerate(messages):
                with st.chat_message(message["role"]):
                    st.markdown(readable_markdown(message["content"]))
                    if message["role"] == "assistant":
                        render_sources(message.get("sources", []), f"chat-{i}")
            with message_control, st.form(f"question-form-{chat_key}", clear_on_submit=True):
                question = st.text_area(
                    "Your question",
                    placeholder="Ask about this PDF…",
                    max_chars=6000,
                    height=100,
                    label_visibility="collapsed",
                    disabled=not ready or not selected,
                )
                submitted = st.form_submit_button(
                    "Ask", type="primary", disabled=not ready or not selected
                )
            if submitted and question.strip():
                with st.chat_message("user"):
                    st.markdown(readable_markdown(question))
                try:
                    with st.spinner("Researching your reports…"):
                        result = ask_equity_question(
                            question, active["index_dir"], messages[-6:], selected
                        )
                    messages.extend(
                        [
                            {"role": "user", "content": question},
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            },
                        ]
                    )
                    st.rerun()
                except Exception as exc:
                    show_error("Research", exc)

st.divider()
st.caption("Answers may contain mistakes. Check the cited pages.")
