import logging

from .runner import EvaluationResults

logger = logging.getLogger(__name__)

REQUIRED_METRICS = ["cer", "rtf", "speaker_similarity", "pas_score"]


def validate_results(results: EvaluationResults) -> bool:
    issues = []

    if not results.results:
        issues.append("No results found in EvaluationResults")

    for r in results.results:
        if not r.success:
            continue
        for metric in REQUIRED_METRICS:
            if metric not in r.metrics:
                issues.append(
                    f"Missing metric '{metric}' in {r.model_name}:{r.sentence_id}"
                )

    success_count = sum(1 for r in results.results if r.success)
    fail_count = len(results.failed_sentences)
    total = results.total_sentences

    if issues:
        for issue in issues[:10]:
            logger.warning(f"Validation: {issue}")
        if len(issues) > 10:
            logger.warning(f"... and {len(issues)-10} more issues")
        return False

    logger.info(
        f"Validation passed: {success_count}/{total} sentences succeeded, "
        f"{fail_count} failed."
    )
    return True
