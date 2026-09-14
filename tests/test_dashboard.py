import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard import (DashboardMetric, Finding, ReportBrief, automatic_changes,
                       load_dashboard, series_groups, validate_brief)


def empty_dashboard(total=2):
    return {**{key: [] for key in ['insights', 'metrics', 'segments', 'risks', 'drivers', 'sources']},
            'analyzed': total, 'total': total, 'errors': []}


def metric(year=2024, value=100):
    return dict(name='Revenue', scope='Consolidated', fiscal_year=year, period='FY', duration_months=12,
        value=value, unit='millions', currency='USD', basis='GAAP', source_id=f'r{year}-S1',
        quote=f'Revenue {value}', period_quote=f'{year}', unit_quote='USD millions')


class DashboardTests(unittest.TestCase):
    def test_changes_are_calculated_from_comparable_values(self):
        changes = automatic_changes(series_groups([metric(), metric(2025, 120)]))
        self.assertEqual(len(changes), 1)
        self.assertIn('20.00%', changes[0]['text'])
        self.assertEqual(changes[0]['source_ids'], ['r2024-S1', 'r2025-S1'])

    def test_growth_rate_change_uses_percentage_points(self):
        rows = [dict(metric(2024, 30), name='Azure revenue growth', unit='percent', currency='N/A'),
                dict(metric(2025, 34), name='Azure revenue growth', unit='percent', currency='N/A')]
        self.assertIn('4.00 percentage points', automatic_changes(series_groups(rows))[0]['text'])

    def test_mixed_periods_currencies_and_bases_never_form_a_trend(self):
        rows = [metric(), dict(metric(2025, 120), duration_months=3, period='Q4'),
                dict(metric(2025, 120), currency='EUR'), dict(metric(2025, 120), basis='non-GAAP')]
        self.assertEqual(automatic_changes(series_groups(rows)), [])

    def test_conflicting_values_are_not_silently_selected(self):
        groups = series_groups([metric(), metric(2024, 105), metric(2025, 120)])
        self.assertEqual(groups[0]['conflicts'], [2024])
        self.assertEqual(automatic_changes(groups), [])

    def test_unit_conversion_and_repeated_comparative_columns(self):
        rows = [metric(), dict(metric(2024, .1), unit='billions'), metric(2025, 120)]
        group = series_groups(rows)[0]
        self.assertEqual(len(group['points']), 2)
        self.assertEqual(group['conflicts'], [])

    def test_unverified_figures_and_findings_are_dropped(self):
        source = {'id': 'r2024-S1', 'content': 'Revenue 100. FY2024. USD millions.'}
        brief = ReportBrief(insights=[Finding(text='Invented', source_id=source['id'], quote='Made up')],
            metrics=[DashboardMetric(**metric(value=200))], segments=[], risks=[], drivers=[])
        verified = validate_brief(brief, [source], {'report_id': 'r2024'})
        self.assertEqual(verified['metrics'], [])
        self.assertEqual(verified['insights'], [])

    def test_verified_metric_retains_source(self):
        source = {'id': 'r2024-S1', 'content': 'Revenue 100. FY2024. USD millions.'}
        brief = ReportBrief(insights=[], metrics=[DashboardMetric(**metric())], segments=[], risks=[], drivers=[])
        result = validate_brief(brief, [source], {'report_id': 'r2024'})
        self.assertEqual(len(result['metrics']), 1)
        self.assertEqual(result['sources'][0]['id'], source['id'])

    def test_cache_only_analyzes_new_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            reports = [{'report_id': 'a', 'fiscal_year': '2024', 'filename': 'a.pdf', 'report_type': 'Annual Report'}]
            path.write_text(json.dumps({'reports': reports}))
            brief = {key: [] for key in ['insights', 'metrics', 'segments', 'risks', 'drivers', 'sources']}
            with patch('dashboard.analyze_report', return_value=brief) as analyze:
                self.assertEqual(load_dashboard(folder)['analyzed'], 1)
                load_dashboard(folder)
                self.assertEqual(analyze.call_count, 1)
                reports.append(dict(reports[0], report_id='b', fiscal_year='2025'))
                path.write_text(json.dumps({'reports': reports}))
                self.assertEqual(load_dashboard(folder)['analyzed'], 2)
                self.assertEqual(analyze.call_count, 2)

    def test_partial_failure_preserves_success_and_can_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            reports = [{'report_id': rid, 'fiscal_year': '2024', 'filename': rid + '.pdf', 'report_type': 'Annual Report'} for rid in ['a', 'b']]
            (Path(folder) / 'metadata.json').write_text(json.dumps({'reports': reports}))
            brief = {key: [] for key in ['insights', 'metrics', 'segments', 'risks', 'drivers', 'sources']}
            with patch('dashboard.analyze_report', side_effect=[brief, ValueError('No verified figures')]):
                result = load_dashboard(folder)
                self.assertEqual(result['analyzed'], 1)
                self.assertEqual(len(result['errors']), 1)
            with patch('dashboard.analyze_report', return_value=brief) as retry:
                self.assertEqual(load_dashboard(folder)['analyzed'], 2)
                self.assertEqual(retry.call_count, 1)

    def test_selected_pdf_excludes_other_saved_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            reports = [{'report_id': rid, 'fiscal_year': '2024', 'filename': rid + '.pdf',
                        'report_type': 'Annual Report'} for rid in ['a', 'b']]
            (Path(folder) / 'metadata.json').write_text(json.dumps({'reports': reports}))
            brief = {key: [] for key in ['insights', 'metrics', 'segments', 'risks', 'drivers', 'sources']}
            with patch('dashboard.analyze_report', return_value=brief) as analyze:
                result = load_dashboard(folder, report_ids=['b'])
                self.assertEqual(result['total'], 1)
                self.assertEqual(analyze.call_count, 1)
                self.assertEqual(analyze.call_args.args[1]['report_id'], 'b')
            with self.assertRaises(ValueError):
                load_dashboard(folder, report_ids=['missing'])
