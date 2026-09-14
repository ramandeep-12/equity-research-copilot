import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ingest import ingest_pdf, read_pdf, list_companies, prepare_report
from report_metadata import ReportMetadata
from research import ResearchAnswer, build_sources, validate_answer, retrieve_evidence, rewrite_follow_up_question
from metrics import quote_has_value


def pdf_bytes(text):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), text)
        return pdf.tobytes()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.detector = patch("ingest.extract_report_metadata")
        self.detect_mock = self.detector.start()
        self.addCleanup(self.detector.stop)
        self.metadata = ReportMetadata(company_name="Example Inc", fiscal_year="2024",
                                       report_type="Annual Report", fiscal_period="FY")

        self.detect_mock.return_value = self.metadata

    def tearDown(self):
        self.temp.cleanup()

    def test_blank_pdf_rejected(self):
        with self.assertRaisesRegex(ValueError, "No readable text"):
            read_pdf(pdf_bytes(""))

    def test_invalid_pdf_rejected(self):
        with self.assertRaises(ValueError):
            read_pdf(b"not a PDF")

    def test_page_number_is_one_based(self):
        pages, total, _ = read_pdf(pdf_bytes("Revenue 123 million"))
        self.assertEqual((pages[0][0], total), (1, 1))

    @patch("ingest.get_embeddings", return_value=DeterministicFakeEmbedding(size=16))
    def test_real_chroma_persistence_dedup_and_company_isolation(self, _):
        content = pdf_bytes("Example Inc FY2024 revenue 123 million")
        first = ingest_pdf(content, "annual.pdf", self.metadata, self.root)
        with patch("ingest.extract_report_metadata", side_effect=AssertionError("Must not call API")):
            duplicate = ingest_pdf(content, "renamed.pdf", root=self.root)
            prepared = prepare_report(content, "renamed.pdf", self.root)
        self.assertTrue(duplicate["reused"])
        self.assertTrue(prepared["reused"])
        self.assertEqual(len(list_companies(self.root)[0]["reports"]), 1)
        other = self.metadata.model_copy(update={"company_name": "Other Inc"})
        self.detect_mock.return_value = other
        second = ingest_pdf(pdf_bytes("Other company revenue 500"), "other.pdf", other, self.root)
        self.assertNotEqual(first["index_dir"], second["index_dir"])
        from langchain_chroma import Chroma
        store = Chroma(collection_name="equity_research", persist_directory=first["index_dir"],
                       embedding_function=DeterministicFakeEmbedding(size=16))
        self.assertEqual(len(store.get()["ids"]), first["chunks"])
        result = store.similarity_search("revenue", k=1)[0]
        self.assertEqual(result.metadata["report_id"], first["report_id"])
        self.assertEqual(result.metadata["page"], 1)

    def test_unknown_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "company") :
            ingest_pdf(pdf_bytes("Text"), "a.pdf", self.metadata.model_copy(update={"company_name": "Unknown"}), self.root)

    def test_failed_index_does_not_publish_manifest(self):
        with patch("ingest.get_embeddings"), patch("ingest.Chroma") as cls:
            cls.return_value.add_documents.side_effect = RuntimeError("Simulated failure")
            with self.assertRaises(RuntimeError):
                ingest_pdf(pdf_bytes("Example revenue 100"), "a.pdf", self.metadata, self.root)
            self.assertEqual(list_companies(self.root), [])
            self.assertEqual(cls.return_value.delete.call_count, 2)

    def test_source_excerpts_on_same_page_are_combined(self):
        docs = [Document(page_content=text, metadata={"report_id": "a", "page": 1}) for text in ["Revenue 100", "Income 20"]]
        sources = build_sources(docs)
        self.assertEqual(len(sources), 1)
        self.assertIn("Income 20", sources[0]["content"])

    def test_invalid_citations_fail_closed(self):
        sources = [{"id": "S1", "report_id": "a", "page": 1}]
        result = validate_answer(ResearchAnswer(answer="Revenue grew [S99]", source_ids=["S99"]), sources)
        self.assertEqual(result["sources"], [])
        self.assertNotIn("Revenue grew", result["answer"])

    def test_valid_citation_keeps_exact_report_page(self):
        result = validate_answer(ResearchAnswer(answer="Revenue 100 [S1]", source_ids=["S1"]),
                                 [{"id": "S1", "report_id": "a", "page": 33}])
        self.assertEqual(result["used_sources"], [{"report_id": "a", "page": 33}])

    def test_retrieval_preserves_both_reports(self):
        (self.root / "metadata.json").write_text(json.dumps({"reports": [{"report_id": "a"}, {"report_id": "b"}]}))
        def search(query, k, filter):
            return [Document(page_content=filter["report_id"], metadata={"report_id": filter["report_id"], "page": 1})]
        with patch("research.get_vector_store") as store, patch("research.generate_search_queries", return_value=["revenue"]):
            store.return_value.similarity_search.side_effect = search
            docs = retrieve_evidence("growth", self.root, ["a", "b"])
        self.assertEqual({d.metadata["report_id"] for d in docs}, {"a", "b"})

    def test_cross_company_report_id_rejected(self):
        (self.root / "metadata.json").write_text(json.dumps({"reports": [{"report_id": "a"}]}))
        with self.assertRaises(ValueError):
            retrieve_evidence("revenue", self.root, ["foreign-id"])

    def test_follow_up_uses_history(self):
        with patch("research.get_llm") as client:
            client.return_value.invoke.return_value = SimpleNamespace(content="Why did revenue grow in FY2024?")
            question = rewrite_follow_up_question("Why?", [{"role": "user", "content": "Revenue FY2024"}])
            self.assertIn("FY2024", question)
            self.assertIn("Revenue FY2024", str(client.return_value.invoke.call_args))

    def test_metric_value_must_match_quote_and_sign(self):
        self.assertTrue(quote_has_value("Revenue 1,250 million", 1250))
        self.assertTrue(quote_has_value("Loss (250)", -250))
        self.assertFalse(quote_has_value("Loss (250)", 250))
        self.assertFalse(quote_has_value("Revenue 30%", 34))


if __name__ == "__main__":
    unittest.main()
