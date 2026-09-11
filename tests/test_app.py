import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_empty_app_without_key(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                "OPENAI_API_KEY": "", "EQUITY_DATA_DIR": root, "EQUITY_WORKSPACE_MODE": "session"}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 4)
            self.assertTrue(any("API" in item.value for item in app.info))
            self.assertTrue(app.button[0].disabled)

    def test_populated_workspace_chat_comparison_and_extraction(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-key", "EQUITY_DATA_DIR": root, "EQUITY_WORKSPACE_MODE": "persistent"}):
            folder = Path(root) / "example"
            folder.mkdir()
            reports = [{"report_id": f"r{i}", "filename": f"FY{2024+i}.pdf", "fiscal_year": str(2024+i),
                        "fiscal_period": "FY", "report_type": "Annual Report", "pages": 2, "chunks": 4} for i in range(2)]
            (folder / "metadata.json").write_text(json.dumps({"company_name": "Example Inc", "reports": reports}))
            result = {"answer": "Revenue grew [S1]", "sources": [{"id": "S1", "report_id": "r0", "page": 1,
                      "source": "FY2024.pdf", "report_type": "Annual Report", "fiscal_year": "2024", "content": "Revenue grew"}]}
            with patch("research.ask_equity_question", return_value=result) as ask:
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
                self.assertFalse(app.exception)
                app.chat_input[0].set_value("How did revenue grow?").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_message), 2)
                app.chat_input[0].set_value("Why?").run()
                self.assertEqual(len(ask.call_args.args[2]), 2)
                app.text_input[0].set_value("revenue").run()
                next(b for b in app.button if b.label == "Analyze changes").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(ask.call_args.kwargs["report_ids"], ["r0", "r1"])
                app.run()
                self.assertTrue(any(m.value == result["answer"] for m in app.markdown))
            with patch("metrics.extract_financial_metrics", return_value=[]):
                next(b for b in app.button if b.label == "Extract cited figures").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.warning), 2)


if __name__ == "__main__":
    unittest.main()
