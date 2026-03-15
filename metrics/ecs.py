import logging
import os
import tempfile
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

TONE_TO_EMOTIONS = {
    "warm": ["Calmness", "Interest", "Sympathy"],
    "neutral": {
        "positive": ["Concentration", "Determination"],
        "negative": ["Excitement", "Joy"],
    },
    "calm": ["Calmness"],
    "urgent": ["Concern", "Empathic Pain"],
}


def _extract_prosody_scores(job_predictions) -> Dict[str, float]:
    scores: Dict[str, list] = {}
    try:
        for file_pred in job_predictions:
            for prediction in file_pred.results.predictions:
                for model_pred in prediction.models.prosody.grouped_predictions:
                    for entry in model_pred.predictions:
                        for emotion in entry.emotions:
                            name = emotion.name
                            score = emotion.score
                            if name not in scores:
                                scores[name] = []
                            scores[name].append(score)
    except Exception as e:
        logger.warning(f"Error extracting prosody scores: {e}")
    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_ecs(
    audio_bytes: bytes,
    intended_tone: str,
    category: str,
) -> Optional[float]:
    api_key = os.environ.get("HUME_API_KEY", "")
    skip_ecs = os.environ.get("SKIP_ECS", "false").lower() == "true"

    if not api_key or skip_ecs:
        logger.info("ECS skipped (no HUME_API_KEY or SKIP_ECS=true)")
        return None

    try:
        from hume import HumeClient
        from hume.expression_measurement.batch import Models, Prosody

        client = HumeClient(api_key=api_key)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            job = client.expression_measurement.batch.start_inference_job_from_local_file(
                file=tmp_path,
                models=Models(prosody=Prosody()),
            )

            for _ in range(30):
                time.sleep(2)
                details = client.expression_measurement.batch.get_job_details(
                    id=job.job_id
                )
                if details.state.status in ("COMPLETED", "FAILED"):
                    break

            if details.state.status != "COMPLETED":
                logger.warning(f"Hume job did not complete: {details.state.status}")
                return None

            predictions = client.expression_measurement.batch.get_job_predictions(
                id=job.job_id
            )
            scores = _extract_prosody_scores(predictions)
        finally:
            os.unlink(tmp_path)

        time.sleep(0.5)

        if intended_tone == "warm":
            emotions = TONE_TO_EMOTIONS["warm"]
            vals = [scores.get(e, 0.0) for e in emotions]
            ecs = sum(vals) / len(vals) if vals else 0.0

        elif intended_tone == "neutral":
            pos = TONE_TO_EMOTIONS["neutral"]["positive"]
            neg = TONE_TO_EMOTIONS["neutral"]["negative"]
            pos_score = sum(scores.get(e, 0.0) for e in pos) / len(pos)
            neg_score = sum(scores.get(e, 0.0) for e in neg) / len(neg)
            ecs = max(0.0, min(1.0, (pos_score - neg_score + 1.0) / 2.0))

        elif intended_tone == "calm":
            ecs = scores.get("Calmness", 0.0)

        elif intended_tone == "urgent":
            emotions = TONE_TO_EMOTIONS["urgent"]
            vals = [scores.get(e, 0.0) for e in emotions]
            ecs = sum(vals) / len(vals) if vals else 0.0

        else:
            logger.warning(f"Unknown intended_tone '{intended_tone}'. Returning 0.")
            ecs = 0.0

        return float(ecs)

    except Exception as e:
        logger.error(f"ECS computation failed: {e}")
        return None


class ECSMetric:
    def __init__(self):
        self.name = "ecs"

    def compute(self, audio_bytes: bytes, intended_tone: str, category: str) -> Dict:
        score = compute_ecs(audio_bytes, intended_tone, category)
        return {"ecs": score}
