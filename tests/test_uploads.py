import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest
from ingest import ingest_pdf, list_companies
from report_metadata import ReportDetection, ReportMetadata, company_key, validate_detection
from research import compare_reports
from upload_queue import merge_pending, validate_batch
from test_pipeline import pdf_bytes
from test_dashboard import empty_dashboard


class Upload:
    def __init__(self, name, data):
        self.name, self.data = name, data

    def getvalue(self):
        return self.data


def metadata(company="Microsoft Corporation", year="2024"):
    return ReportMetadata(company_name=company, fiscal_year=year, report_type="Annual Report", fiscal_period="FY")


def pending(company, rid):
    return {"detail": {**metadata(company).model_dump(), "report_id": rid}, "bytes": b"test"}


class UploadValidationTests(unittest.TestCase):
    def test_queue_preserves_first_report_and_deduplicates(self):
        a, b = pending("Microsoft", "a"), pending("Microsoft Corporation", "b")
        queue = merge_pending([a], [a, b])
        self.assertEqual([p['detail']['report_id'] for p in queue], ["a", "b"])
        self.assertEqual(validate_batch(queue), "microsoft")

    def test_mixed_company_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "different companies"):
            validate_batch([pending("Microsoft", "a"), pending("Apple", "b")])

    def test_other_company_cannot_enter_active_library(self):
        with self.assertRaisesRegex(ValueError, "active company"):
            validate_batch([pending("Apple", "b")], active_company="Microsoft")

    def test_no_fuzzy_merging(self):
        self.assertEqual(company_key("Microsoft Corporation"), company_key("MICROSOFT"))
        self.assertNotEqual(company_key("ABC Holdings Inc"), company_key("ABC Inc"))

    def test_random_document_and_fabricated_evidence_rejected(self):
        detection = ReportDetection(**metadata().model_dump(), is_financial_report=False,
            reason="Resume", issuer_quote="Microsoft", financial_quote="2024")
        with self.assertRaisesRegex(ValueError, "not a supported"):
            validate_detection(detection, "Microsoft 2024 resume")
        detection.is_financial_report = True
        detection.financial_quote = "Revenue was 200 million"
        with self.assertRaisesRegex(ValueError, "could not be verified"):
            validate_detection(detection, "Microsoft 2024 resume")

    def test_verified_report_accepted(self):
        detection = ReportDetection(**metadata().model_dump(), is_financial_report=True,
            reason="Annual results", issuer_quote="Microsoft Corporation", financial_quote="Revenue 200 million")
        self.assertEqual(validate_detection(detection, "Microsoft Corporation\nRevenue 200 million").fiscal_year, "2024")

    def test_manual_metadata_cannot_bypass_detection(self):
        with tempfile.TemporaryDirectory() as root, patch("ingest.extract_report_metadata", side_effect=ValueError("Not a financial report")):
            with self.assertRaises(ValueError):
                ingest_pdf(pdf_bytes("A random essay"), "essay.pdf", metadata(), root)
            self.assertFalse(list_companies(root))

    def test_upload_opens_chat_and_removes_uploader(self):
        a = Upload("2024.pdf", pdf_bytes("Microsoft FY2024 revenue 200 million"))
        b = Upload("2025.pdf", pdf_bytes("Microsoft Corporation FY2025 revenue 220 million"))
        def detect(data):
            return metadata("Microsoft", "2024") if data == a.data else metadata("Microsoft Corporation", "2025")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-key", "EQUITY_DATA_DIR": root, "EQUITY_WORKSPACE_MODE": "persistent"}), \
                patch("ingest.extract_report_metadata", side_effect=detect), \
                patch("ingest.get_embeddings", return_value=DeterministicFakeEmbedding(size=16)), \
                patch("dashboard.load_dashboard", return_value=empty_dashboard()) as analyze, \
                patch("streamlit.file_uploader", return_value=a) as uploader:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(list_companies(root)[0]['reports']), 1)
            self.assertEqual(analyze.call_count, 1)
            calls_after_upload = uploader.call_count
            uploader.return_value = b
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(uploader.call_count, calls_after_upload)
            self.assertEqual(len(list_companies(root)[0]['reports']), 1)
            self.assertEqual(analyze.call_count, 1)
            self.assertEqual(len(app.text_area), 1)
            self.assertEqual(len(app.multiselect), 0)
            self.assertFalse(uploader.call_args.kwargs['accept_multiple_files'])

    def test_failed_single_upload_can_retry(self):
        upload = Upload("broken.pdf", b"not a pdf")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-key", "EQUITY_DATA_DIR": root, "EQUITY_WORKSPACE_MODE": "persistent"}), \
                patch("streamlit.file_uploader", return_value=upload):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertTrue(any(button.label == "Retry upload" for button in app.button))
            self.assertEqual(list_companies(root), [])

    def test_comparison_requires_both_reports_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            reports = [{**metadata().model_dump(), "report_id": rid, "filename": rid + ".pdf"} for rid in ['a', 'b']]
            (Path(root) / 'metadata.json').write_text(json.dumps({'company_name': 'Microsoft', 'reports': reports}))
            with patch('research.ask_equity_question', return_value={'answer': 'Growth was 10%', 'sources': [{'report_id': 'a'}]}):
                result = compare_reports('revenue', root, 'a', 'b')
                self.assertIn('could not be verified', result['answer'])
                self.assertNotIn('Growth was 10%', result['answer'])
            with self.assertRaises(ValueError):
                compare_reports('revenue', root, 'a', 'a')
