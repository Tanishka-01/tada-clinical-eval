import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .runner import EvaluationResults, SentenceResult

logger = logging.getLogger(__name__)


def save_results(results: EvaluationResults, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "results": [asdict(r) for r in results.results],
        "models_evaluated": results.models_evaluated,
        "metrics_computed": results.metrics_computed,
        "total_sentences": results.total_sentences,
        "failed_sentences": results.failed_sentences,
        "timestamp": results.timestamp,
        "duration_seconds": results.duration_seconds,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


def load_results(path: str) -> EvaluationResults:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = EvaluationResults(
        models_evaluated=data["models_evaluated"],
        metrics_computed=data["metrics_computed"],
        total_sentences=data["total_sentences"],
        failed_sentences=data["failed_sentences"],
        timestamp=data["timestamp"],
        duration_seconds=data["duration_seconds"],
    )
    for r in data["results"]:
        sr = SentenceResult(
            sentence_id=r["sentence_id"],
            model_name=r["model_name"],
            text=r["text"],
            category=r["category"],
            subcategory=r.get("subcategory", ""),
            intended_tone=r.get("intended_tone", "neutral"),
            metrics=r.get("metrics", {}),
            audio_path=r.get("audio_path"),
            inference_time=r.get("inference_time", 0.0),
            success=r.get("success", True),
            error_message=r.get("error_message"),
        )
        results.results.append(sr)
    return results


def save_csv(results: EvaluationResults, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results.results:
        row = {
            "sentence_id": r.sentence_id,
            "model_name": r.model_name,
            "text": r.text,
            "category": r.category,
            "subcategory": r.subcategory,
            "intended_tone": r.intended_tone,
            "inference_time": r.inference_time,
            "success": r.success,
        }
        for k, v in r.metrics.items():
            if not isinstance(v, (list, dict)):
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"CSV saved to {path}")


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path))
