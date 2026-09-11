"""Equity Research Copilot — Streamlit entry point."""
import csv
import hashlib
import io
import logging
import os
import re
from pathlib import Path
import uuid

import streamlit as st
from ingest import ingest_pdf, list_companies, prepare_report
from research import ask_equity_question, citation_label
from metrics import METRICS, extract_financial_metrics
from settings import data_root

st.set_page_config(page_title="Equity Research Copilot", page_icon="◈", layout="wide")

# Streamlit Cloud secrets and local .env both work. Never display the key.
try:
    for setting in ("OPENAI_API_KEY", "OPENAI_MODEL", "EQUITY_DATA_DIR", "EQUITY_WORKSPACE_MODE"):
        if setting in st.secrets:
            os.environ[setting] = str(st.secrets[setting])
except FileNotFoundError:
    pass

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:#f7f8fa;color:#182c30;}
.block-container {max-width:1280px;padding-top:2.4rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;}
h1 {font-weight:800!important;font-size:2.5rem!important;}
[data-testid="stSidebar"] {background:#102e2b;}
[data-testid="stSidebar"] * {color:#e4eee9;}
[data-testid="stSidebar"] input {color:#182c30;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#182c30;}
[data-testid="stSidebar"] button {background:#20443e;border-color:#375850;}
[data-testid="stSidebar"] hr {border-color:#35544d;}
[data-testid="stMetric"] {background:white;border:1px solid #e1e8e5;border-radius:12px;padding:18px 22px;}
[data-testid="stMetricLabel"] {color:#71817c;font-size:.8rem;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-weight:700;}
[data-testid="stChatMessage"] {background:white;border:1px solid #e1e8e5;border-radius:12px;}
.stButton button[kind="primary"] {background:#21765d;border-color:#21765d;border-radius:8px;}
[data-testid="stExpander"] {background:white;border-radius:10px;}
[data-baseweb="tab-list"] {gap:26px;border-bottom:1px solid #dce5e1;margin:12px 0 22px;}
[data-baseweb="tab"] {font-weight:600;}
.eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.16em;color:#4e8572;margin-bottom:10px;}
.hero {padding:35px;border:1px solid #d8e6df;border-radius:16px;background:linear-gradient(120deg,#edf5ef,#fff);margin:20px 0 28px;}
.hero h2 {font-size:2.05rem;margin:0 0 12px;max-width:620px;}
.hero p {color:#60766c;max-width:650px;line-height:1.7;}
.step {font-size:.72rem;letter-spacing:.12em;color:#508770;font-weight:700;}
.brand {font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;line-height:1.3;margin:8px 0 10px;}
.side-caption {color:#9bb6ab!important;font-size:.85rem;line-height:1.6;}
.footer {color:#83928c;font-size:.75rem;margin-top:35px;border-top:1px solid #dfe7e2;padding-top:16px;}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("workspace_id", uuid.uuid4().hex)
persistent = os.getenv("EQUITY_WORKSPACE_MODE", "session") == "persistent"
root = data_root() if persistent else data_root() / "sessions" / st.session_state.workspace_id
ready = bool(os.getenv("OPENAI_API_KEY"))


def show_error(action, error):
    logging.getLogger(__name__).error("%s failed (%s)", action, type(error).__name__)
    if isinstance(error, ValueError):
        st.error(str(error))
    else:
        st.error(f"{action} could not finish. Check your API credentials, quota and connection, then retry. "
                 "Your previously indexed reports are still available.")


def report_label(report):
    return (f"FY{report['fiscal_year']} · {report.get('fiscal_period', 'Unknown')} · "
            f"{report['report_type']} · {report['filename']}")


def render_sources(sources, key):
    if sources:
        st.caption(f"{len(sources)} supporting source(s) · page numbers refer to the PDF, including its cover")
    for source in sources:
        with st.expander(f"[{source.get('id', 'Source')}] {citation_label(source)}"):
            st.text(source["content"])
            pdf = Path(active["index_dir"]) / f"{source['report_id']}.pdf"
            if pdf.exists():
                st.download_button("Download source report", pdf.read_bytes(),
                                   file_name=source.get("source", "report.pdf"), mime="application/pdf",
                                   key=f"{key}-{source['id']}-pdf")


def readable_markdown(text):
    """Keep currency dollar signs literal instead of opening LaTeX math spans."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def export_answer(result):
    sources = "\n\n".join(f"[{s.get('id', 'Source')}] {citation_label(s)}\n{s['content']}"
                             for s in result.get("sources", []))
    return readable_markdown(result["answer"]) + "\n\n## Sources\n\n" + readable_markdown(sources)


try:
    companies = list_companies(root)
except Exception as exc:
    show_error("Loading the report library", exc)
    companies = []

with st.sidebar:
    st.markdown('<div class="eyebrow">RESEARCH WORKSPACE</div><div class="brand">◈ Equity Research<br>Copilot</div>', unsafe_allow_html=True)
    st.markdown('<p class="side-caption">From financial disclosures<br>to defensible insights.</p>', unsafe_allow_html=True)
    st.divider()
    if companies:
        company_path = st.selectbox("Active company", [c["index_dir"] for c in companies],
            format_func=lambda path: next(c["company_name"] for c in companies if c["index_dir"] == path),
            key="company_picker")
        active = next(c for c in companies if c["index_dir"] == company_path)
    else:
        active = None
        st.caption("Your companies will appear here after indexing.")
    st.divider()
    st.markdown("**Workspace status**")
    st.caption("● API configured" if ready else "○ API setup needed")
    st.caption("Persistent library" if persistent else "Private session library")
    st.caption("Reports stay on this server." if persistent else "Reports are isolated to this browser session. A new session starts a new library.")
    with st.expander("Setup & methodology"):
        st.write("Set OPENAI_API_KEY in .env locally or in Streamlit deployment secrets.")
        st.write("Report text is sent to OpenAI for metadata, embeddings and analysis. PDF page numbers include the cover.")
        st.write("Searchable PDFs only, up to 50 MB each. Confirm detected metadata before indexing.")
        st.write("Financial extraction verifies quoted figures; review accounting units and period definitions before interpreting trends.")

st.markdown('<div class="eyebrow">DOCUMENT INTELLIGENCE / EQUITY RESEARCH</div>', unsafe_allow_html=True)
st.title(active["company_name"] if active else "Better research starts with evidence.")
st.caption("Explore the disclosures. Connect the periods. Trace every insight to its source.")

if not ready:
    st.info("Add OPENAI_API_KEY to your .env file or Streamlit secrets to enable report analysis. The workspace is ready to explore.")

reports = active["reports"] if active else []
cols = st.columns(4)
for col, label, value in zip(cols, ["REPORTS INDEXED", "FISCAL YEARS", "PAGES OF EVIDENCE", "KNOWLEDGE BASES"],
                            [len(reports), len({r['fiscal_year'] for r in reports}),
                             sum(r.get('pages', 0) or 0 for r in reports), len(companies)]):
    col.metric(label, value)

library, research_tab, compare_tab, metrics_tab = st.tabs(["Report library", "Research chat", "Compare periods", "Financial metrics"])

with library:
    if not reports:
        st.markdown('<div class="hero"><div class="eyebrow">YOUR ANALYST WORKBENCH</div><h2>One company. Multiple reports.<br>A clearer investment picture.</h2><p>Bring annual, quarterly and earnings reports together. Ask follow-up questions, compare disclosures across periods, and inspect the evidence behind each answer.</p></div>', unsafe_allow_html=True)
        for col, number, title, description in zip(st.columns(3), ["01 / COLLECT", "02 / UNDERSTAND", "03 / RESEARCH"],
            ["Build your report library", "Review the report identity", "Follow the evidence"],
            ["Upload searchable financial PDFs. Duplicate content is recognized automatically.",
             "Confirm the company, fiscal year and reporting period before indexing.",
             "Get conversational answers and period comparisons with exact source pages."]):
            with col:
                st.markdown(f'<div class="step">{number}</div>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(description)
        st.divider()
    st.subheader("Add financial reports")
    files = st.file_uploader("Annual reports, quarterly filings or earnings releases", type="pdf", accept_multiple_files=True)
    st.caption("Upload → Detect metadata → Review → Index. Add reports without clearing your existing research.")
    signature = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files)
    if st.session_state.get("pending_signature") != signature:
        st.session_state.pop("pending_reports", None)
        st.session_state["pending_signature"] = signature
    if st.button("Detect report details", type="primary", disabled=not files or not ready):
        pending = []
        progress = st.progress(0, text="Reading report details…")
        for i, file in enumerate(files):
            try:
                detail = prepare_report(file.getvalue(), file.name, root)
                pending.append({"detail": detail, "bytes": file.getvalue()})
            except Exception as exc:
                st.caption(file.name)
                show_error("Report detection", exc)
            progress.progress((i + 1) / len(files))
        progress.empty()
        st.session_state.pending_reports = pending
    pending = st.session_state.get("pending_reports", [])
    if pending:
        st.markdown("**Review detected details**")
        st.caption("Use the same company name for all periods of an issuer. Correct any detection errors here.")
        with st.form("review_reports"):
            edited = []
            for i, item in enumerate(pending):
                detail = item["detail"]
                st.markdown(f"**{detail['filename']}** · {detail['pages']} pages")
                if detail["reused"]:
                    st.info(f"Already indexed for {detail['company_name']} — duplicate skipped.")
                    edited.append(None)
                    continue
                c1, c2, c3, c4 = st.columns([3, 1, 2, 1])
                company = c1.text_input("Company", detail["company_name"], key=f"co-{detail['report_id']}-{i}")
                year = c2.text_input("Fiscal year", detail["fiscal_year"], key=f"yr-{detail['report_id']}-{i}")
                types = ["Annual Report", "Quarterly Report", "Earnings Report", "Other"]
                report_type = c3.selectbox("Report type", types, index=types.index(detail["report_type"]) if detail["report_type"] in types else 3, key=f"ty-{i}")
                periods = ["FY", "Q1", "Q2", "Q3", "Q4", "Unknown"]
                period = c4.selectbox("Period", periods, index=periods.index(detail.get("fiscal_period", "Unknown")) if detail.get("fiscal_period") in periods else 5, key=f"pe-{i}")
                edited.append(dict(company_name=company, fiscal_year=year.strip(), report_type=report_type, fiscal_period=period))
            submitted = st.form_submit_button("Confirm & index reports", type="primary", disabled=not ready)
        if submitted:
            failures, new_count, reused_count = [], 0, 0
            with st.status("Building the company knowledge base…", expanded=True) as status:
                for item, metadata in zip(pending, edited):
                    try:
                        st.write(f"Processing {item['detail']['filename']}")
                        result = ingest_pdf(item["bytes"], item["detail"]["filename"], metadata, root)
                        reused_count += int(result["reused"])
                        new_count += int(not result["reused"])
                    except Exception as exc:
                        failures.append(item)
                        show_error("Indexing", exc)
                status.update(label=f"{new_count} indexed · {reused_count} duplicates skipped · {len(failures)} failed", state="error" if failures else "complete")
            st.session_state.pending_reports = failures
            if not failures:
                st.session_state["library_notice"] = f"{new_count} new report(s) indexed. {reused_count} duplicate(s) skipped."
                st.rerun()
    if st.session_state.get("library_notice"):
        st.success(st.session_state.pop("library_notice"))
    if reports:
        st.subheader("Indexed reports")
        for report in reports:
            with st.expander(report_label(report)):
                st.caption(f"{report.get('pages', '—')} pages · {report.get('chunks', '—')} searchable passages · ID {report['report_id']}")
                pdf = Path(active["index_dir"]) / f"{report['report_id']}.pdf"
                if pdf.exists():
                    st.download_button("Download original PDF", pdf.read_bytes(), report["filename"], "application/pdf", key=f"library-{report['report_id']}")

with research_tab:
    st.subheader("Ask a better follow-up.")
    st.caption("Research across reports, then keep the conversation going. Answers use your selected evidence only.")
    if not active:
        st.info("Index your first company report in the Report library to start researching.")
    else:
        by_id = {r["report_id"]: r for r in reports}
        selected = st.multiselect("Evidence scope · up to 12 reports", list(by_id), default=list(by_id)[:12],
                                 format_func=lambda rid: report_label(by_id[rid]), key=f"scope-{active['index_dir']}")
        chat_key = "chat-" + active["index_dir"] + "-" + "-".join(sorted(selected))
        messages = st.session_state.setdefault(chat_key, [])
        if messages:
            c1, c2 = st.columns([1, 4])
            if c1.button("Clear conversation"):
                st.session_state[chat_key] = []
                st.rerun()
            c2.download_button("Export research notes", "\n\n".join(
                f"## {m['role'].title()}\n\n" + (export_answer({"answer": m['content'], "sources": m.get('sources', [])}))
                for m in messages), "research-notes.md", "text/markdown")
        else:
            st.caption("Try: How did revenue growth change? • What drove operating margins? • Which risks became more significant?")
        for i, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.markdown(readable_markdown(message["content"]))
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []), f"chat-{i}")
        question = st.chat_input("Ask about growth, margins, cash flow or risks…", disabled=not ready or not selected or len(selected) > 12)
        if question:
            with st.chat_message("user"):
                st.markdown(readable_markdown(question))
            try:
                with st.spinner("Reading across your selected reports…"):
                    result = ask_equity_question(question, active["index_dir"], messages[-6:], selected)
                messages.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": result["answer"], "sources": result["sources"]}])
                st.rerun()
            except Exception as exc:
                show_error("Research", exc)

with compare_tab:
    st.subheader("Understand what changed.")
    st.caption("Select two reports and a research topic. Compare figures, disclosed drivers and implications side by side.")
    if len(reports) < 2:
        st.info("Index at least two reports for the same company to compare periods.")
    else:
        options = {r["report_id"]: r for r in reports}
        c1, c2 = st.columns(2)
        left = c1.selectbox("Baseline report", list(options), format_func=lambda rid: report_label(options[rid]), key=f"left-{active['index_dir']}")
        right = c2.selectbox("Comparison report", [rid for rid in options if rid != left], format_func=lambda rid: report_label(options[rid]), key=f"right-{active['index_dir']}")
        topic = st.text_input("Focus of comparison", placeholder="e.g. Azure growth, operating margins, capital expenditure")
        comparison_key = "comparison-" + active["index_dir"] + left + right + topic
        if st.button("Analyze changes", type="primary", disabled=not ready or not topic.strip()):
            try:
                with st.spinner("Comparing evidence from both reports…"):
                    result = ask_equity_question(
                        f"Compare {topic}. Baseline: {report_label(options[left])}. "
                        f"Comparison: {report_label(options[right])}. Explain values in each report, "
                        "the size and direction of the change, disclosed drivers, and key takeaway.",
                        active["index_dir"], report_ids=[left, right])
                st.session_state[comparison_key] = result
            except Exception as exc:
                show_error("Comparison", exc)
        if comparison_key in st.session_state:
            result = st.session_state[comparison_key]
            st.markdown(readable_markdown(result["answer"]))
            render_sources(result["sources"], "comparison")
            st.download_button("Export comparison", export_answer(result), "report-comparison.md", "text/markdown")

with metrics_tab:
    st.subheader("Turn disclosures into structured data.")
    st.caption("Extract a consolidated financial metric from each selected report, with a verifiable excerpt for every value.")
    if not reports:
        st.info("Add reports to extract financial metrics.")
    else:
        metric = st.selectbox("Financial metric", METRICS)
        metric_options = {r["report_id"]: r for r in reports}
        metric_reports = st.multiselect("Reports to extract · up to 12", list(metric_options), default=list(metric_options)[:12],
                format_func=lambda rid: report_label(metric_options[rid]), key=f"metrics-scope-{active['index_dir']}")
        metric_key = "metrics-" + active["index_dir"] + metric + "-".join(metric_reports)
        if st.button("Extract cited figures", type="primary", disabled=not ready or not metric_reports or len(metric_reports) > 12):
            rows, missing = [], []
            with st.status("Extracting and checking reported figures…"):
                for rid in metric_reports:
                    try:
                        extracted = extract_financial_metrics(active["index_dir"], rid, metric)
                        rows.extend(extracted)
                        if not extracted:
                            missing.append(report_label(metric_options[rid]))
                    except Exception as exc:
                        missing.append(report_label(metric_options[rid]))
                        show_error("Financial extraction", exc)
            st.session_state[metric_key] = {"rows": rows, "missing": missing}
        if metric_key in st.session_state:
            extraction = st.session_state[metric_key]
            rows = extraction["rows"]
            for label in extraction["missing"]:
                st.warning(f"No verified figure returned for {label}. Try Research chat to inspect the disclosure.")
            if rows:
                columns = ["metric", "period", "value", "unit", "currency", "basis", "report", "page"]
                st.dataframe([{k: row[k] for k in columns} for row in rows], hide_index=True, width="stretch")
                st.caption("Values preserve the reported scale. Periods and accounting bases may differ; review the evidence before comparison.")
                for i, row in enumerate(rows):
                    with st.expander(f"{row['report']} · {row['period']} · PDF page {row['page']}"):
                        st.text(row["quote"])
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                # Neutralize spreadsheet formula prefixes in text fields.
                writer.writerows({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                                  for k, v in row.items()} for row in rows)
                st.download_button("Download financial metrics CSV", output.getvalue(), "financial-metrics.csv", "text/csv")

st.markdown('<div class="footer">◈ Equity Research Copilot &nbsp; / &nbsp; Source-grounded financial intelligence &nbsp; / &nbsp; Verify cited disclosures before making investment decisions.</div>', unsafe_allow_html=True)
