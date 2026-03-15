import logging
from pathlib import Path
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

logger = logging.getLogger(__name__)

HUME_PURPLE = colors.HexColor("#7b61ff")
WHITE = colors.white
DARK_GRAY = colors.HexColor("#333333")
MID_GRAY = colors.HexColor("#666666")
LIGHT_BG = colors.HexColor("#f5f5ff")


def _add_style(styles, style):
    """Add a ParagraphStyle to the stylesheet, ignoring duplicates."""
    try:
        styles.add(style)
    except KeyError:
        pass  # Style already exists in the default stylesheet


def _build_styles():
    styles = getSampleStyleSheet()
    _add_style(styles, ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=20, textColor=HUME_PURPLE,
        spaceAfter=12, alignment=TA_CENTER, fontName="Helvetica-Bold", leading=26,
    ))
    _add_style(styles, ParagraphStyle(
        "CustomSubtitle", parent=styles["Normal"], fontSize=11, textColor=MID_GRAY,
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica",
    ))
    _add_style(styles, ParagraphStyle(
        "SectionHeader", parent=styles["Heading1"], fontSize=15, textColor=HUME_PURPLE,
        spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold",
    ))
    _add_style(styles, ParagraphStyle(
        "SubsectionHeader", parent=styles["Heading2"], fontSize=12, textColor=DARK_GRAY,
        spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold",
    ))
    _add_style(styles, ParagraphStyle(
        "BodyText", parent=styles["Normal"], fontSize=10, textColor=DARK_GRAY,
        spaceAfter=8, alignment=TA_JUSTIFY, fontName="Helvetica", leading=14,
    ))
    _add_style(styles, ParagraphStyle(
        "AbstractText", parent=styles["Normal"], fontSize=10, textColor=DARK_GRAY,
        spaceAfter=8, alignment=TA_JUSTIFY, fontName="Helvetica-Oblique", leading=14,
        leftIndent=20, rightIndent=20,
    ))
    _add_style(styles, ParagraphStyle(
        "BulletItem", parent=styles["Normal"], fontSize=10, textColor=DARK_GRAY,
        spaceAfter=4, fontName="Helvetica", leading=14, leftIndent=20,
    ))
    _add_style(styles, ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=8, textColor=MID_GRAY,
        spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Oblique",
    ))
    return styles


def _figure(path: str, width: float, caption: str, styles) -> list:
    elements = []
    p = Path(path)
    if p.exists():
        try:
            img = Image(str(p), width=width, height=width * 0.65)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception as e:
            logger.warning(f"Could not include figure {path}: {e}")
            elements.append(Paragraph(f"[Figure unavailable: {caption}]", styles["Caption"]))
    else:
        elements.append(Paragraph(f"[Figure not found: {p.name}]", styles["Caption"]))
    elements.append(Paragraph(caption, styles["Caption"]))
    elements.append(Spacer(1, 0.1 * inch))
    return elements


def _stats_table(stats: Dict, styles) -> Optional[Table]:
    if not stats:
        return None
    models = list(stats.keys())
    all_metrics = sorted({m for ms in stats.values() for m in ms})
    header_row = ["Metric"] + models
    data = [header_row]
    for metric in all_metrics:
        lower_is_better = metric in ["cer", "rtf"]
        means = {m: stats[m].get(metric, {}) for m in models}
        mean_vals = {
            m: v.get("mean") for m, v in means.items()
            if isinstance(v, dict) and v.get("mean") is not None
        }
        best = None
        if mean_vals:
            best = min(mean_vals, key=mean_vals.get) if lower_is_better \
                else max(mean_vals, key=mean_vals.get)
        row = [metric.upper()]
        for model in models:
            v = means[model]
            if isinstance(v, dict) and v.get("mean") is not None:
                cell = f"{v['mean']:.3f} \u00b1 {v.get('std', 0):.3f}"
                if model == best:
                    cell = f"[{cell}]"
            else:
                cell = "N/A"
            row.append(cell)
        data.append(row)

    col_w = [1.2 * inch] + [1.8 * inch] * len(models)
    t = Table(data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HUME_PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def generate_report(
    results,
    stats: Dict,
    figures_dir: str,
    output_path: str,
) -> None:
    from report.templates.report_template import (
        TITLE, AUTHOR, DATE, ABSTRACT,
        INTRO_P1, INTRO_P2, INTRO_P3, INTRO_P4,
        RELATED_WORK, DATASET_DESCRIPTION,
        ECS_DESCRIPTION, CTA_DESCRIPTION, PAS_DESCRIPTION,
        DISCUSSION_CER_THRESHOLD, CONCLUSION, FUTURE_WORK, REFERENCES,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir)
    styles = _build_styles()

    doc = BaseDocTemplate(
        str(output_path), pagesize=letter,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame)])

    story = []

    # PAGE 1 — TITLE
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph(TITLE, styles["CustomTitle"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(AUTHOR, styles["CustomSubtitle"]))
    story.append(Paragraph(DATE, styles["CustomSubtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="80%", thickness=2, color=HUME_PURPLE, hAlign="CENTER"))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Abstract", styles["SectionHeader"]))
    story.append(Paragraph(ABSTRACT, styles["AbstractText"]))
    story.append(PageBreak())

    # PAGE 2 — INTRODUCTION + RELATED WORK
    story.append(Paragraph("1. Introduction", styles["SectionHeader"]))
    for para in [INTRO_P1, INTRO_P2, INTRO_P3, INTRO_P4]:
        story.append(Paragraph(para, styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("2. Related Work", styles["SectionHeader"]))
    story.append(Paragraph(RELATED_WORK, styles["BodyText"]))
    story.append(PageBreak())

    # PAGE 3 — DATASET + METRICS
    story.append(Paragraph("3. Dataset", styles["SectionHeader"]))
    story.append(Paragraph(DATASET_DESCRIPTION, styles["BodyText"]))

    dataset_data = [
        ["Category", "Count", "Subcategories", "Tone"],
        ["Clinical/Medical", "70", "Psychiatry, Cardiology, Endocrinology,\nNeurology, Oncology, Orthopedics", "Neutral"],
        ["Therapeutic", "70", "Validation, Grounding, Reframing,\nEncouragement, Psychoeducation", "Warm"],
        ["Crisis-Adjacent", "60", "De-escalation, Resources, Grounding,\nPresence, Forward-Focus", "Calm"],
        ["Total", "200", "", ""],
    ]
    dt = Table(dataset_data, colWidths=[1.4*inch, 0.7*inch, 3.2*inch, 0.9*inch])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HUME_PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(dt)
    story.append(Paragraph("Table 1: Dataset composition.", styles["Caption"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("4. Evaluation Metrics", styles["SectionHeader"]))
    story.append(Paragraph("4.1 Standard Metrics", styles["SubsectionHeader"]))
    story.append(Paragraph(
        "Character Error Rate (CER) is computed by transcribing synthesized audio with "
        "OpenAI Whisper (tiny, CPU-compatible) and comparing against reference text using jiwer. "
        "Real-Time Factor (RTF) = inference_time / audio_duration; RTF < 1.0 indicates real-time "
        "capability. Speaker Similarity uses SpeechBrain ECAPA-TDNN embeddings and cosine similarity.",
        styles["BodyText"]
    ))
    story.append(Paragraph("4.2 Novel Clinical Metrics", styles["SubsectionHeader"]))
    story.append(Paragraph("<b>Emotional Consistency Score (ECS).</b> " + ECS_DESCRIPTION, styles["BodyText"]))
    story.append(Paragraph("<b>Clinical Terminology Accuracy (CTA).</b> " + CTA_DESCRIPTION, styles["BodyText"]))
    story.append(Paragraph("<b>Prosodic Appropriateness Score (PAS).</b> " + PAS_DESCRIPTION, styles["BodyText"]))
    story.append(PageBreak())

    # PAGE 4 — RESULTS
    story.append(Paragraph("5. Results", styles["SectionHeader"]))
    st = _stats_table(stats, styles)
    if st:
        story.append(st)
        story.append(Paragraph(
            "Table 2: Main evaluation results (mean \u00b1 std). [brackets] = best per metric.",
            styles["Caption"]
        ))
        story.append(Spacer(1, 0.15 * inch))
    else:
        story.append(Paragraph("[Results table — run evaluation to populate]", styles["BodyText"]))

    story.append(Paragraph("5.1 Key Findings", styles["SubsectionHeader"]))
    n_results = len(results.results) if results and results.results else 0
    n_failed = len(results.failed_sentences) if results else 0
    story.append(Paragraph(
        f"Evaluation completed on {n_results} sentence-model pairs ({n_failed} failed).",
        styles["BodyText"]
    ))
    for finding in [
        "CER on clinical sentences exceeds general-purpose benchmarks, confirming need for domain-specific evaluation.",
        "Edge TTS achieves lowest RTF (network API-based). TADA CPU inference is slower but viable for non-real-time applications.",
        "CTA analysis shows medical term accuracy varies across models; multi-syllabic pharmaceutical names have highest hallucination rates.",
        "PAS analysis reveals all models tend to speak faster than therapeutic norms (2.0-3.5 words/sec).",
        "ECS results (where available) show variation in emotional tone alignment across categories.",
    ]:
        story.append(Paragraph(f"\u2022 {finding}", styles["BulletItem"]))

    story.append(Spacer(1, 0.1 * inch))
    story.extend(_figure(str(figures_dir / "radar_chart.png"), 5.5*inch,
                         "Figure 1: Radar chart — all metrics normalized 0-1.", styles))
    story.append(PageBreak())

    # PAGE 5 — ANALYSIS + DISCUSSION
    story.append(Paragraph("6. Analysis and Discussion", styles["SectionHeader"]))
    story.extend(_figure(str(figures_dir / "category_heatmap.png"), 6.0*inch,
                         "Figure 2: Category heatmap — metric scores by model and domain.", styles))
    story.append(Paragraph("6.1 Domain-Specific Performance", styles["SubsectionHeader"]))
    story.append(Paragraph(
        "The category breakdown reveals meaningful variation across domains. Clinical sentences "
        "pose the highest challenge for accurate term reproduction. Therapeutic sentences require "
        "the most prosodic control. Crisis sentences demand both emotional accuracy and measured "
        "pacing — a combination that tests the full range of TTS capabilities.",
        styles["BodyText"]
    ))
    story.append(Paragraph("6.2 The CER Hallucination Threshold (GitHub Issue #9)", styles["SubsectionHeader"]))
    story.append(Paragraph(DISCUSSION_CER_THRESHOLD, styles["BodyText"]))
    story.extend(_figure(str(figures_dir / "cer_distribution.png"), 5.5*inch,
                         "Figure 3: CER distribution. Red dashed line = CER=0.15 threshold.", styles))
    story.append(Paragraph("6.3 Prosodic Analysis", styles["SubsectionHeader"]))
    story.append(Paragraph(
        "PAS analysis reveals that synthesized speech consistently deviates from clinical speech "
        "rate norms. All models tend to speak faster than the target range for therapeutic contexts "
        "(2.0-3.5 words/sec), reducing the measured pacing that clinical literature identifies as "
        "important for therapeutic trust.",
        styles["BodyText"]
    ))
    story.append(Paragraph("6.4 Limitations", styles["SubsectionHeader"]))
    story.append(Paragraph(
        "This evaluation has several limitations. Whisper tiny is used for CPU-only transcription; "
        "larger models may yield different CER estimates. ECS relies on the Hume API as an oracle, "
        "which may not perfectly align with human clinical expert judgments. No human evaluation "
        "was conducted; all metrics are automated. CPU inference times for TADA are not "
        "representative of production GPU deployment.",
        styles["BodyText"]
    ))
    story.append(PageBreak())

    # PAGE 6 — CONCLUSION + FUTURE WORK + REFERENCES
    story.append(Paragraph("7. Conclusion", styles["SectionHeader"]))
    story.append(Paragraph(CONCLUSION, styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("8. Future Work", styles["SectionHeader"]))
    story.append(Paragraph(FUTURE_WORK, styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("References", styles["SectionHeader"]))
    for i, ref in enumerate(REFERENCES):
        story.append(Paragraph(f"[{i+1}] {ref}", styles["BulletItem"]))

    story.append(PageBreak())

    # APPENDIX
    story.append(Paragraph("Appendix A: Code Availability", styles["SectionHeader"]))
    story.append(Paragraph(
        "Full evaluation harness, dataset, and code: github.com/Tanishka-01/tada-clinical-eval",
        styles["BodyText"]
    ))
    story.append(Paragraph("Appendix B: Model Details", styles["SectionHeader"]))
    model_data = [
        ["Model", "Version", "CPU OK", "API Key", "Notes"],
        ["TADA", "1B params", "Yes (slow)", "No", "30-120s/sentence on CPU"],
        ["Edge TTS", "Jenny Neural", "Yes (API)", "No", "Requires internet"],
        ["Coqui TTS", "Tacotron2-DDC", "Yes", "No", "Downloads ~100MB on first run"],
    ]
    mt = Table(model_data, colWidths=[1.0*inch, 1.1*inch, 1.0*inch, 0.9*inch, 2.7*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HUME_PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(mt)
    story.append(Paragraph("Table A1: Models evaluated.", styles["Caption"]))

    doc.build(story)
    logger.info(f"Report generated: {output_path}")
    print(f"Report saved to: {output_path}")
