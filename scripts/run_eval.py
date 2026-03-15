#!/usr/bin/env python3
"""Main evaluation entry point — runs the full TADA clinical speech evaluation."""

import argparse
import logging
import sys
from datetime import datetime
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


def parse_args():
    parser = argparse.ArgumentParser(description="TADA Clinical Speech Evaluation Harness")
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated: tada,edge-tts,coqui (default: all)"
    )
    parser.add_argument(
        "--categories", default="all",
        help="Comma-separated: clinical,therapeutic,crisis (default: all)"
    )
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--skip-ecs", action="store_true",
                        help="Skip ECS metric (use if no HUME_API_KEY)")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--report-path", default="./tada_clinical_eval_report.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("TADA Clinical Speech Evaluation Harness")
    logger.info("=" * 60)

    from dataset.sentences import SENTENCES
    sentences = SENTENCES
    if args.categories != "all":
        cats = [c.strip() for c in args.categories.split(",")]
        sentences = [s for s in SENTENCES if s["category"] in cats]
    logger.info(f"Loaded {len(sentences)} sentences")

    from models.edge_tts_model import EdgeTTSModel
    from models.coqui_model import CoquiTTSModel
    from models.tada_model import TADAModel

    all_models = {"tada": TADAModel(), "edge-tts": EdgeTTSModel(), "coqui": CoquiTTSModel()}
    model_list = list(all_models.values()) if args.models == "all" else [
        all_models[m] for m in args.models.split(",") if m.strip() in all_models
    ]

    available = [m for m in model_list if m.is_available()]
    for m in model_list:
        if m.is_available():
            logger.info(f"Model available: {m.name}")
        else:
            logger.warning(f"Model NOT available (skipping): {m.name}")

    if not available:
        logger.error("No models available.")
        sys.exit(1)

    from evaluation.runner import EvaluationRunner
    runner = EvaluationRunner(output_dir=args.output_dir)
    logger.info(f"Starting: {len(available)} models x {len(sentences)} sentences")

    results = runner.run_full_evaluation(
        sentences=sentences,
        models=available,
        metrics_config={"cer": True, "rtf": True, "speaker_similarity": True,
                        "pas": True, "cta": True, "ecs": not args.skip_ecs},
        max_sentences=args.max_sentences,
        skip_ecs=args.skip_ecs,
    )

    from evaluation.storage import save_results, save_csv, load_csv
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.output_dir) / f"results_{ts}.json"
    csv_path = Path(args.output_dir) / f"results_{ts}.csv"
    save_results(results, str(results_path))
    save_csv(results, str(csv_path))

    from analysis.statistics import compute_summary_stats
    df = load_csv(str(csv_path))
    stats = compute_summary_stats(df)

    from analysis.charts import generate_all_charts
    figs = generate_all_charts(df, args.figures_dir)
    logger.info(f"Generated {len(figs)} figures")

    from report.generator import generate_report
    generate_report(results, stats, args.figures_dir, args.report_path)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results JSON : {results_path}")
    logger.info(f"Results CSV  : {csv_path}")
    logger.info(f"Report PDF   : {args.report_path}")
    n_f = len(results.failed_sentences)
    logger.info(f"Success rate : {results.total_sentences - n_f}/{results.total_sentences}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
