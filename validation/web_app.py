"""可视化 Web 界面：独立 Flask 应用 + 可挂载 Blueprint。

两种用法：
  * 独立运行：`python -m validation.cli serve --port 5001`
  * 挂到现有 web_interface：`app.register_blueprint(validation_blueprint, url_prefix="/validation")`

该模块只读取 `validation_runs/` 下的结果，不参与、也不阻塞线上决策流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, Flask, jsonify, redirect, render_template, request, send_file, url_for

from .config import BacktestConfig
from .runner import list_runs, load_run, resolve_runs_dir, run_backtest

validation_blueprint = Blueprint(
    "validation",
    __name__,
    template_folder="templates",
    static_folder=None,
)


def _summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = payload.get("metrics", {})
    return {
        "run_id": payload.get("run_id"),
        "strategy": payload.get("strategy"),
        "data": {
            "asset_name": payload.get("data", {}).get("asset_name"),
            "first": payload.get("data", {}).get("first"),
            "last": payload.get("data", {}).get("last"),
            "rows": payload.get("data", {}).get("rows"),
        },
        "stats": payload.get("stats"),
        "composite": m.get("composite", {}).get("score"),
        "accuracy": m.get("accuracy", {}).get("accuracy"),
        "total_return_pct": m.get("performance", {}).get("total_return_pct"),
        "benchmark_return_pct": m.get("performance", {}).get("benchmark_total_return_pct"),
        "max_drawdown_pct": m.get("risk", {}).get("max_drawdown_pct"),
        "p95_latency_ms": m.get("timeliness", {}).get("p95_decision_latency_ms"),
        "anomalies": {
            "total_flags": payload.get("anomalies", {}).get("total_flags"),
            "flagged_decisions": payload.get("anomalies", {}).get("flagged_decisions"),
            "by_category": payload.get("anomalies", {}).get("by_category"),
        },
        "decisions_hash": payload.get("decisions_hash"),
        "reproducible": payload.get("reproducible"),
        "report_url": url_for("validation.report_page", run_id=payload.get("run_id")),
    }


def _config_from_request(data: Dict[str, Any]) -> BacktestConfig:
    def _num(key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            return default

    return BacktestConfig(
        asset=str(data.get("asset") or "COMEX"),
        timeframe=str(data.get("timeframe") or "1d"),
        source=data.get("source") or None,
        start=data.get("start") or None,
        end=data.get("end") or None,
        run_id=str(data.get("run_id") or "web_default"),
        incremental=bool(data.get("incremental")),
        strategy=str(data.get("strategy") or "rule"),
        horizon_bars=_int("horizon_bars", 1),
        warmup_bars=_int("warmup_bars", 30),
        fee_pct=_num("fee_pct", 0.0),
        slippage_pct=_num("slippage_pct", 0.0),
        use_stops=bool(data.get("use_stops", True)),
        stop_atr_mult=_num("stop_atr_mult", 2.0),
        risk_reward=_num("risk_reward", 1.5),
        benchmark=str(data.get("benchmark") or "buy_hold"),
        recorded_path=data.get("recorded_path") or None,
        min_confidence=_num("min_confidence", 0.35),
    )


@validation_blueprint.route("/")
def index():
    return render_template("validation.html")


@validation_blueprint.route("/api/health")
def health():
    return jsonify({"ok": True, "module": "decision-validation"})


@validation_blueprint.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(silent=True) or {}
    try:
        cfg = _config_from_request(data)
        mode = str(data.get("data_mode") or "live")
        csv_path = data.get("csv_path") or None
        synthetic: Optional[Dict[str, Any]] = None
        if mode == "synthetic":
            synthetic = {
                "n": max(60, int(data.get("synthetic_bars") or 320)),
                "seed": int(data.get("synthetic_seed") or 7),
            }
        elif mode == "csv" and not csv_path:
            return jsonify({"ok": False, "error": "data_mode=csv 时必须填写 csv_path"}), 400
        elif mode == "live":
            csv_path = None

        payload = run_backtest(cfg, csv_path=csv_path, synthetic=synthetic)
        return jsonify({"ok": True, "summary": _summary(payload)})
    except Exception as exc:  # noqa: BLE001 - 统一返回可读错误
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400


@validation_blueprint.route("/api/runs")
def api_runs():
    return jsonify({"ok": True, "runs": list_runs()})


@validation_blueprint.route("/api/result/<run_id>")
def api_result(run_id: str):
    try:
        return jsonify({"ok": True, "result": load_run(run_id)})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@validation_blueprint.route("/api/decisions/<run_id>")
def api_decisions(run_id: str):
    try:
        payload = load_run(run_id)
        return jsonify({"ok": True, "decisions": payload.get("decisions", []), "trades": payload.get("trades", [])})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@validation_blueprint.route("/report/<run_id>")
def report_page(run_id: str):
    path = resolve_runs_dir() / run_id / "report.html"
    if not path.exists():
        return jsonify({"ok": False, "error": f"报告不存在：{run_id}"}), 404
    return send_file(path, mimetype="text/html")


def create_app() -> Flask:
    """独立应用（默认挂载在 /validation）。"""
    app = Flask(__name__)
    app.register_blueprint(validation_blueprint, url_prefix="/validation")

    @app.route("/")
    def _root():
        return redirect(url_for("validation.index"))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)
