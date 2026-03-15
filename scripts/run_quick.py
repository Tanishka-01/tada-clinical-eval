#!/usr/bin/env python3
"""Quick eval on 20 sentences — tests the full pipeline in ~10 minutes."""

import logging
import sys
import datetime
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("TADA Clinical Eval — QUICK MODE (20 sentences)")
    logger.info("Models: TADA + edge-tts | ECS: skipped")
    logger.info("=" * 60)

    from dataset.sentences import SENTENCES

    clinical = [s for s in SENTENCES if s["category"] == "clinical"][:7]
    therapeutic = [s for s in SENTENCES if s["category"] == "therapeutic"][:7]
    crisis = [s for s in SENTENCES if s["category"] == "crisis"][:6]
    sentences = clinical + therapeutic + crisis
    logger.info(
        f"Sampled {len(sentences)} sentences "
        f"(clinical={len(clinical)}, therapeutic={len(therapeutic)}, crisis={len(crisis)})"
    )

    from models.tada_model import TADAModel
    from models.edge_tts_model import EdgeTTSModel

    models = []
    for model in [TADAModel(), EdgeTTSModel()]:
        if model.is_available():
            models.append(model)
            logger.info(f"Using: {model.name}")
        else:
            logger.warning(f"Not available (skipping): {model.name}")

    if not models:
        logger.error("No models available.")
        sys.exit(1)

    output_dir = "./results/quick"
    figures_dir = "./figures/quick"

    from evaluation.runner import EvaluationRunner
    runner = EvaluationRunner(output_dir=output_dir)
    results = runner.run_full_evaluation(
        sentences=sentences,
        models=models,
        metrics_config={"cer": True, "rtf": True, "pas": True},
        skip_ecs=True,
    )

    from evaluation.storage import save_results, save_csv, load_csv
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = f"{output_dir}/quick_results_{ts}.json"
    csv_path = f"{output_dir}/quick_results_{ts}.csv"
    save_results(results, results_path)
    save_csv(results, csv_path)

    from analysis.statistics import compute_summary_stats
    df = load_csv(csv_path)
    stats = compute_summary_stats(df)

    from analysis.charts import generate_all_charts
    generate_all_charts(df, figures_dir)

    report_path = f"{output_dir}/quick_report_{ts}.pdf"
    from report.generator import generate_report
    generate_report(results, stats, figures_dir, report_path)

    logger.info("\n" + "=" * 60)
    logger.info("QUICK EVAL COMPLETE")
    logger.info(f"Results : {results_path}")
    logger.info(f"Report  : {report_path}")
    n_f = len(results.failed_sentences)
    logger.info(f"Success : {results.total_sentences - n_f}/{results.total_sentences}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
