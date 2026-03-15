import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SentenceResult:
    sentence_id: str
    model_name: str
    text: str
    category: str
    subcategory: str
    intended_tone: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    audio_path: Optional[str] = None
    inference_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EvaluationResults:
    results: List[SentenceResult] = field(default_factory=list)
    models_evaluated: List[str] = field(default_factory=list)
    metrics_computed: List[str] = field(default_factory=list)
    total_sentences: int = 0
    failed_sentences: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_seconds: float = 0.0


class EvaluationRunner:
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_evaluation(
        self,
        sentences: List[Dict],
        models: list,
        metrics_config: Dict,
        max_sentences: Optional[int] = None,
        skip_ecs: bool = False,
    ) -> EvaluationResults:
        from metrics.cer import CERMetric
        from metrics.rtf import RTFMetric
        from metrics.speaker_similarity import SpeakerSimilarityMetric
        from metrics.ecs import ECSMetric
        from metrics.cta import CTAMetric
        from metrics.pas import PASMetric

        if max_sentences:
            sentences = sentences[:max_sentences]

        eval_results = EvaluationResults(
            models_evaluated=[m.name for m in models],
            metrics_computed=list(metrics_config.keys()),
            total_sentences=len(sentences) * len(models),
            timestamp=datetime.utcnow().isoformat(),
        )

        cer_metric = CERMetric()
        rtf_metric = RTFMetric()
        pas_metric = PASMetric()
        cta_metric = CTAMetric()
        ecs_metric = ECSMetric() if not skip_ecs else None
        speaker_metric = SpeakerSimilarityMetric()

        overall_start = time.perf_counter()
        completed = 0

        for model in models:
            if not model.is_available():
                logger.warning(f"Model {model.name} is not available. Skipping.")
                continue

            model_dir = self.output_dir / model.name
            model_dir.mkdir(parents=True, exist_ok=True)

            ref_audio_bytes = None

            logger.info(f"\n{'='*60}\nEvaluating model: {model.name}\n{'='*60}")

            for i, sentence in enumerate(tqdm(sentences, desc=f"{model.name}")):
                sid = sentence["id"]
                text = sentence["text"]
                category = sentence["category"]
                subcategory = sentence.get("subcategory", "")
                intended_tone = sentence.get("intended_tone", "neutral")
                medical_terms = sentence.get("medical_terms", [])

                logger.info(f"Evaluating {model.name} on {sid} ({i+1}/{len(sentences)})")

                result = SentenceResult(
                    sentence_id=sid,
                    model_name=model.name,
                    text=text,
                    category=category,
                    subcategory=subcategory,
                    intended_tone=intended_tone,
                )

                try:
                    audio_bytes, inference_time = model.generate(text)
                    result.inference_time = inference_time

                    audio_path = model_dir / f"{sid}.wav"
                    audio_path.write_bytes(audio_bytes)
                    result.audio_path = str(audio_path)

                    if ref_audio_bytes is None:
                        ref_audio_bytes = audio_bytes
                        speaker_metric.set_reference(ref_audio_bytes)

                    metrics_dict: Dict[str, Any] = {}

                    try:
                        cer_result = cer_metric.compute(audio_bytes, text, medical_terms)
                        metrics_dict.update(cer_result)
                    except Exception as e:
                        logger.warning(f"CER failed for {sid}: {e}")
                        metrics_dict["cer"] = None

                    try:
                        rtf_result = rtf_metric.compute(audio_bytes, inference_time)
                        metrics_dict.update(rtf_result)
                    except Exception as e:
                        logger.warning(f"RTF failed for {sid}: {e}")
                        metrics_dict["rtf"] = None

                    try:
                        sim_result = speaker_metric.compute(audio_bytes)
                        metrics_dict.update(sim_result)
                    except Exception as e:
                        logger.warning(f"Speaker similarity failed for {sid}: {e}")
                        metrics_dict["speaker_similarity"] = None

                    try:
                        pas_result = pas_metric.compute(audio_bytes, category, text)
                        metrics_dict["pas_score"] = pas_result.get("pas_score")
                        metrics_dict["pas_features"] = pas_result.get("features", {})
                    except Exception as e:
                        logger.warning(f"PAS failed for {sid}: {e}")
                        metrics_dict["pas_score"] = None

                    if medical_terms:
                        try:
                            cta_result = cta_metric.compute(audio_bytes, text, medical_terms)
                            metrics_dict["cta_score"] = cta_result.get("cta_score")
                            metrics_dict["cta_incorrect_terms"] = cta_result.get("incorrect_terms", [])
                        except Exception as e:
                            logger.warning(f"CTA failed for {sid}: {e}")
                            metrics_dict["cta_score"] = None

                    if ecs_metric is not None:
                        try:
                            ecs_result = ecs_metric.compute(audio_bytes, intended_tone, category)
                            metrics_dict.update(ecs_result)
                        except Exception as e:
                            logger.warning(f"ECS failed for {sid}: {e}")
                            metrics_dict["ecs"] = None

                    result.metrics = metrics_dict

                except MemoryError as e:
                    logger.error(f"Memory error on {sid}: {e}")
                    result.success = False
                    result.error_message = str(e)
                    eval_results.failed_sentences.append(f"{model.name}:{sid}")
                except Exception as e:
                    logger.error(f"Error on {sid} with {model.name}: {e}")
                    result.success = False
                    result.error_message = str(e)
                    eval_results.failed_sentences.append(f"{model.name}:{sid}")

                eval_results.results.append(result)
                completed += 1

                if completed % 10 == 0:
                    self._save_intermediate(eval_results)
                    logger.info(f"Saved intermediate results ({completed} processed)")

        eval_results.duration_seconds = time.perf_counter() - overall_start
        return eval_results

    def _save_intermediate(self, results: EvaluationResults) -> None:
        path = self.output_dir / "intermediate_results.json"
        data = {
            "results": [asdict(r) for r in results.results],
            "models_evaluated": results.models_evaluated,
            "metrics_computed": results.metrics_computed,
            "total_sentences": results.total_sentences,
            "failed_sentences": results.failed_sentences,
            "timestamp": results.timestamp,
            "duration_seconds": results.duration_seconds,
        }
        with open(path, "w") as f:
            json.dump(data, f, default=str)
