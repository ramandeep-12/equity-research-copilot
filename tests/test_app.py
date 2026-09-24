import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from test_dashboard import empty_dashboard, metric

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class AppTests(unittest.TestCase):
    def test_empty_app_without_key(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "", "EQUITY_DATA_DIR": root, "EQUITY_WORKSPACE_MODE": "session"},
            ),
        ):
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(len(app.text_area), 0)
            self.assertTrue(any("unavailable" in item.value for item in app.info))

    def test_dashboard_automatically_loads_and_chat_keeps_context(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "EQUITY_DATA_DIR": root,
                    "EQUITY_WORKSPACE_MODE": "persistent",
                },
            ),
        ):
            folder = Path(root) / "example"
            folder.mkdir()
            reports = [
                {
                    "report_id": f"r{i}",
                    "filename": f"FY{2024 + i}.pdf",
                    "fiscal_year": str(2024 + i),
                    "fiscal_period": "FY",
                    "report_type": "Annual Report",
                    "pages": 2,
                    "chunks": 4,
                }
                for i in range(2)
            ]
            (folder / "metadata.json").write_text(
                json.dumps({"company_name": "Example Inc", "reports": reports})
            )
            data = empty_dashboard()
            data["metrics"] = [
                metric(),
                metric(2025, 120),
                dict(metric(2025, 245122), name="Net income"),
            ]
            data["sources"] = [
                {
                    "id": f"r{year}-S1",
                    "report_id": f"r{year - 2024}",
                    "page": 1,
                    "source": f"FY{year}.pdf",
                    "report_type": "Annual Report",
                    "fiscal_year": str(year),
                    "fiscal_period": "FY",
                    "content": f"Revenue {value}. FY{year}. USD millions.",
                }
                for year, value in [(2024, 100), (2025, 120)]
            ]
            data["risks"] = [
                {
                    "text": "Competition is a disclosed risk.",
                    "source_id": "r2024-S1",
                    "quote": "Competition",
                }
            ]
            answer = {"answer": "Revenue grew [S1]", "sources": [dict(data["sources"][0], id="S1")]}
            with (
                patch("dashboard.load_dashboard", return_value=data) as analyze,
                patch("research.ask_equity_question", return_value=answer) as ask,
            ):
                app = AppTest.from_file(APP, default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertEqual(analyze.call_count, 1)
                self.assertTrue(any("245,122.00 millions" in text.value for text in app.markdown))
                self.assertEqual(len(app.metric), 0)
                self.assertFalse(any(button.label == "Analyze changes" for button in app.button))
                self.assertTrue(any(title.value == "Key insights" for title in app.subheader))
                self.assertTrue(any(title.value == "Financial trends" for title in app.subheader))
                self.assertTrue(any("20.00%" in text.value for text in app.markdown))
                app.text_area[0].set_value("How did revenue change?")
                next(button for button in app.button if button.label == "Ask").click().run()
                self.assertEqual(len(app.chat_input), 0)
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_message), 2)
                self.assertEqual(ask.call_args.args[3], ["r0"])
                self.assertEqual(analyze.call_args.kwargs["report_ids"], ["r0"])
                app.text_area[0].set_value("Why?")
                next(button for button in app.button if button.label == "Ask").click().run()
                self.assertEqual(len(ask.call_args.args[2]), 2)
                self.assertEqual(analyze.call_count, 1)
                self.assertFalse(app.exception)
