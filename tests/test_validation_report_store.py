"""报表 / 持久化 / Web 接口测试：交付物完整性与边界容错。"""

import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validation.config import BacktestConfig
from validation.data import synthetic_bars
from validation.report import render_html, render_markdown
from validation.runner import list_runs, load_run, run_backtest


class TestReportAndStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic_bars(150, seed=5)

    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BacktestConfig(run_id="store", warmup_bars=30, min_bars=60)
            payload = run_backtest(cfg, bars=self.bars, runs_dir=tmp, persist=True)

            run_dir = os.path.join(tmp, "store")
            for name in ("manifest.json", "decisions.jsonl", "result.json", "report.html", "report.md"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, name)), name)

            loaded = load_run("store", runs_dir=tmp)
            self.assertEqual(len(loaded["decisions"]), payload["stats"]["decisions"])
            self.assertEqual(loaded["decisions_hash"], payload["decisions_hash"])
            self.assertTrue(os.path.exists(loaded["files"]["report_html"]))

            runs = list_runs(runs_dir=tmp)
            self.assertEqual([r["run_id"] for r in runs], ["store"])

    def test_html_report_sections(self):
        payload = run_backtest(
            BacktestConfig(run_id="html", warmup_bars=30), bars=self.bars, persist=False
        )
        html = render_html(payload)
        for keyword in ("决策正确性验证报告", "决策准确率", "混淆矩阵", "异常决策自动标记",
                        "复现信息", "决策明细", "最大回撤"):
            self.assertIn(keyword, html)

    def test_html_escapes_untrusted_text(self):
        payload = run_backtest(
            BacktestConfig(run_id="escape", warmup_bars=30), bars=self.bars, persist=False
        )
        payload["data"]["asset_name"] = "<script>alert(1)</script>"
        html = render_html(payload)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_empty_payload_renders(self):
        html = render_html(
            {
                "run_id": "empty",
                "engine_version": "1.0.0",
                "created_at": "2025-01-01 00:00:00",
                "strategy": "rule",
                "config": {},
                "data": {},
                "stats": {},
                "metrics": {},
                "anomalies": {},
                "series": {},
                "decisions": [],
                "trades": [],
                "decisions_hash": "x",
            }
        )
        self.assertIn("决策正确性验证报告", html)
        self.assertIn("暂无", html)
        md = render_markdown({})
        self.assertIn("决策正确性验证报告", md)

    def test_ratio_rendered_as_percentage(self):
        """比率型指标（0-1）必须按百分比展示，而不是 0.57%。"""
        payload = {
            "run_id": "pct", "engine_version": "1.0.0", "created_at": "2025-01-01 00:00:00",
            "strategy": "rule", "config": {}, "data": {}, "stats": {}, "series": {},
            "decisions": [], "trades": [], "decisions_hash": "x", "anomalies": {},
            "metrics": {
                "accuracy": {"accuracy": 0.5, "balanced_accuracy": 0.5,
                             "long_accuracy": 0.5, "short_accuracy": 0.5,
                             "evaluated_trades": 10, "wrong": 5},
                "performance": {"total_return_pct": 1.0, "win_rate": 0.6},
                "risk": {}, "timeliness": {}, "composite": {"score": 50},
            },
        }
        html = render_html(payload)
        self.assertIn("50.00%", html)
        self.assertIn("60.00%", html)
        self.assertNotIn("0.50%", html)

    def test_result_json_is_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_backtest(BacktestConfig(run_id="json"), bars=self.bars, runs_dir=tmp, persist=True)
            path = os.path.join(tmp, "json", "result.json")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("metrics", data)
            self.assertIn("accuracy", data["metrics"])
            self.assertNotIn("decisions", data)  # 决策明细单独存 JSONL

    def test_web_endpoints(self):
        from validation.web_app import create_app

        client = create_app().test_client()
        self.assertEqual(client.get("/validation/api/health").status_code, 200)
        page = client.get("/validation/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("决策正确性验证", page.get_data(as_text=True))
        self.assertEqual(client.get("/validation/report/not_exists").status_code, 404)

    def test_web_run_validation_error_json(self):
        from validation.web_app import create_app

        client = create_app().test_client()
        resp = client.post("/validation/api/run", json={"run_id": "bad id!", "data_mode": "synthetic"})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["ok"])


if __name__ == "__main__":
    unittest.main()
