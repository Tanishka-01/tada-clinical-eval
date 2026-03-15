#!/usr/bin/env python3
"""Standalone report generation from saved results JSON."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate report from saved results")
    parser.add_argument("--results-json", required=True, help="Path to results JSON")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--report-path", default="./tada_clinical_eval_report.pdf")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()

    logger.info(f"Loading results from {args.results_json}")
    from evaluation.storage import load_results, save_csv, load_csv
    results = load_results(args.results_json)
    logger.info(f"Loaded {len(results.results)} sentence results")

    csv_path = args.results_json.replace(".json", ".csv")
    save_csv(results, csv_path)
    df = load_csv(csv_path)

    from analysis.statistics import compute_summary_stats
    stats = compute_summary_stats(df)

    if args.regenerate_figures:
        logger.info("Regenerating figures...")
        from analysis.charts import generate_all_charts
        generated = generate_all_charts(df, args.figures_dir)
        logger.info(f"Generated {len(generated)} figures")

    logger.info("Generating report PDF...")
    from report.generator import generate_report
    generate_report(results, stats, args.figures_dir, args.report_path)
    logger.info(f"Report saved: {args.report_path}")


if __name__ == "__main__":
    main()
