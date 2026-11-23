"""运行编排与结果持久化。

职责：
  1. 取数 → 回放 → 指标 → 异常 → 报表 的一站式编排；
  2. `validation_runs/<run_id>/` 落盘（manifest / decisions.jsonl / result.json / report）；
  3. 增量回测：复用同 run_id 下与当前决策指纹一致的已算决策；
  4. 可复现校验：同数据同参数重跑两次，决策哈希必须一致。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anomalies import detect_anomalies
from .config import ENGINE_VERSION, BacktestConfig
from .data import PROJECT_ROOT, load_bars
from .engine import BacktestEngine, BacktestRun
from .metrics import compute_metrics
from .report import write_artifacts
from .strategies import DecisionStrategy, build_strategy

DEFAULT_RUNS_DIR = PROJECT_ROOT / "validation_runs"
RUN_FILES = {
    "manifest": "manifest.json",
    "decisions": "decisions.jsonl",
    "result": "result.json",
    "report": "report.html",
    "markdown": "report.md",
}


def resolve_runs_dir(runs_dir: Optional[str | Path] = None) -> Path:
    path = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _load_decision_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            key = item.pop("key", None)
            if key:
                cache[key] = item
    return cache


def _write_decision_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    lines = []
    for key, decision in cache.items():
        record = {"key": key, **decision}
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))


def _build_payload(
    run: BacktestRun,
    decisions_hash: str,
    decision_fingerprint: str,
    created_at: str,
    timings: Dict[str, float],
    reproducible: bool,
) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "engine_version": ENGINE_VERSION,
        "created_at": created_at,
        "strategy": run.strategy_name,
        "config": run.config.to_dict(),
        "data": run.data_meta,
        "stats": run.stats,
        "timings": timings,
        "decisions_hash": decisions_hash,
        "decision_fingerprint": decision_fingerprint,
        "reproducible": reproducible,
        "metrics": run.metrics,
        "anomalies": {k: v for k, v in run.anomalies.items() if k != "all"},
        "series": run.series,
        "trades": [t.to_dict() for t in run.trades],
    }


def run_backtest(
    config: Optional[BacktestConfig] = None,
    *,
    bars=None,
    data_meta: Optional[Dict[str, Any]] = None,
    strategy: Optional[DecisionStrategy] = None,
    runs_dir: Optional[str | Path] = None,
    csv_path: Optional[str] = None,
    synthetic: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """执行一次完整验证，返回结构化结果 payload（可直接 JSON 序列化）。"""
    cfg = config or BacktestConfig()
    started = time.perf_counter()

    if bars is None:
        bars, data_meta = load_bars(cfg, csv_path=csv_path, synthetic=synthetic)
    data_meta = data_meta or {}
    strategy_obj = strategy or build_strategy(cfg)

    fingerprint_extra = strategy_obj.fingerprint_extra()
    decision_fingerprint = cfg.decision_fingerprint(fingerprint_extra)

    run_dir = resolve_runs_dir(runs_dir) / cfg.run_id
    cached: Dict[str, Dict[str, Any]] = {}
    cached_fingerprint = None
    if persist:
        manifest = _read_json(run_dir / RUN_FILES["manifest"])
        cached_fingerprint = manifest.get("decision_fingerprint")
        if cfg.incremental and cached_fingerprint == decision_fingerprint:
            cached = _load_decision_cache(run_dir / RUN_FILES["decisions"])

    engine = BacktestEngine(cfg, strategy_obj)
    run = engine.run(bars, data_meta=data_meta, cached_decisions=cached)

    metrics_started = time.perf_counter()
    run.metrics = compute_metrics(run)
    run.anomalies = detect_anomalies(run)
    metrics_s = time.perf_counter() - metrics_started

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timings = {
        **run.timings,
        "metrics_s": round(metrics_s, 3),
    }
    reproducible = not bool(getattr(strategy_obj, "nondeterministic", False))
    payload = _build_payload(
        run, run.decisions_hash, decision_fingerprint, created_at, timings, reproducible
    )
    payload["decisions"] = [d.to_dict() for d in run.decisions]

    if persist:
        report_started = time.perf_counter()
        write_artifacts(payload, run_dir)
        timings["report_s"] = round(time.perf_counter() - report_started, 3)
        timings["total_s"] = round(time.perf_counter() - started, 3)

        manifest = {
            "run_id": run.run_id,
            "engine_version": ENGINE_VERSION,
            "created_at": created_at,
            "strategy": run.strategy_name,
            "decision_fingerprint": decision_fingerprint,
            "decisions_hash": run.decisions_hash,
            "data": {
                "mode": data_meta.get("mode"),
                "source": data_meta.get("source"),
                "fingerprint": data_meta.get("fingerprint"),
                "rows": data_meta.get("rows"),
                "first": data_meta.get("first"),
                "last": data_meta.get("last"),
                "attempts": data_meta.get("attempts"),
            },
            "config": cfg.to_dict(),
            "stats": run.stats,
            "timings": timings,
            "reproducible": reproducible,
            "files": RUN_FILES,
        }
        _atomic_write(run_dir / RUN_FILES["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2))
        _write_decision_cache(run_dir / RUN_FILES["decisions"], run.decision_cache)
        _atomic_write(
            run_dir / RUN_FILES["result"],
            json.dumps({k: v for k, v in payload.items() if k != "decisions"}, ensure_ascii=False, indent=2),
        )

    return payload


def load_run(run_id: str, runs_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """读取已持久化的回测结果（含决策明细）。"""
    run_dir = resolve_runs_dir(runs_dir) / run_id
    result = _read_json(run_dir / RUN_FILES["result"])
    if not result:
        raise FileNotFoundError(f"未找到回测结果：{run_dir}")
    cache = _load_decision_cache(run_dir / RUN_FILES["decisions"])
    decisions = [v for v in cache.values()]
    decisions.sort(key=lambda d: d.get("index", 0))
    result["decisions"] = decisions
    result["files"] = {
        "dir": str(run_dir),
        "report_html": str(run_dir / RUN_FILES["report"]),
        "report_markdown": str(run_dir / RUN_FILES["markdown"]),
        "decisions_jsonl": str(run_dir / RUN_FILES["decisions"]),
        "manifest": str(run_dir / RUN_FILES["manifest"]),
    }
    return result


def list_runs(runs_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    root = resolve_runs_dir(runs_dir)
    if not root.exists():
        return []
    runs = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        manifest = _read_json(child / RUN_FILES["manifest"])
        if not manifest:
            continue
        runs.append(
            {
                "run_id": manifest.get("run_id", child.name),
                "created_at": manifest.get("created_at"),
                "strategy": manifest.get("strategy"),
                "asset": (manifest.get("config") or {}).get("asset"),
                "timeframe": (manifest.get("config") or {}).get("timeframe"),
                "decisions_hash": manifest.get("decisions_hash"),
                "data": manifest.get("data"),
                "stats": manifest.get("stats"),
            }
        )
    return runs


def check_reproducible(
    config: BacktestConfig,
    *,
    bars=None,
    data_meta: Optional[Dict[str, Any]] = None,
    csv_path: Optional[str] = None,
    synthetic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """同数据 + 同参数连跑两次，比较决策序列哈希。"""
    first = run_backtest(
        config, bars=bars, data_meta=data_meta, csv_path=csv_path,
        synthetic=synthetic, persist=False,
    )
    second = run_backtest(
        config, bars=bars, data_meta=data_meta, csv_path=csv_path,
        synthetic=synthetic, persist=False,
    )
    same = first["decisions_hash"] == second["decisions_hash"]
    return {
        "reproducible": bool(same and first.get("reproducible", False)),
        "first_hash": first["decisions_hash"],
        "second_hash": second["decisions_hash"],
        "decisions": len(first.get("decisions", [])),
        "strategy": first.get("strategy"),
        "note": "LLM 策略为非确定性，默认不参与可复现断言" if not first.get("reproducible") else "",
    }
