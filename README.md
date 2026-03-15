# TADA Clinical Speech Evaluation Harness

## What This Is

A systematic evaluation of Hume AI's TADA TTS model on clinical and therapeutic speech — the first domain-specific benchmark for healthcare voice AI. This harness provides a complete pipeline from dataset to metrics to publication-ready PDF report.

## Why It Matters

- TADA is evaluated on LibriTTS and EARS — neither represents clinical speech
- Healthcare is Hume's key vertical but no clinical benchmark exists
- CER thresholds from general TTS research may be too permissive for medical contexts (addresses [GitHub issue #9](https://github.com/HumeAI/tada/issues/9) on HumeAI/tada)
- Even small transcription errors in clinical contexts — a mispronounced drug name, an incorrect dosage — carry real consequences

## Dataset

200 sentences across 3 categories, each annotated with medical terminology, intended emotional tone, expected speech rate, and domain subcategory.

| Category | Count | Subcategories |
|----------|-------|---------------|
| Clinical/Medical | 70 | Psychiatry, Cardiology, Endocrinology, Neurology, Oncology, Orthopedics, General |
| Therapeutic | 70 | Validation, Grounding, Reframing, Encouragement, Psychoeducation, Breathing |
| Crisis-Adjacent | 60 | De-escalation, Resources, Grounding, Presence, Forward-Focus |

## Novel Metrics

### 1. Emotional Consistency Score (ECS)
Uses Hume's Expression Measurement API as an oracle for intended emotional tone alignment. Measures whether TADA's synthesized speech *sounds* emotionally appropriate — warm for therapy, neutral for clinical, calm for crisis.

### 2. Clinical Terminology Accuracy (CTA)
CER focused specifically on medical terms. Tests whether models hallucinate, mispronounce, or omit medication names, diagnostic terms, and clinical abbreviations. A CTA < 0.95 is a deployment concern.

### 3. Prosodic Appropriateness Score (PAS)
librosa-based prosodic analysis measuring speech rate, pitch variation, and pause ratio against domain-specific clinical norms derived from clinical speech literature.

## Models Evaluated

| Model | Type | CPU Compatible | Notes |
|-------|------|----------------|-------|
| TADA-1B | Hume AI (open source) | Yes (slow) | 30–120s/sentence on CPU |
| edge-tts | Microsoft (free API) | Yes | Requires internet |
| Coqui TTS | Open source | Yes | Downloads ~100MB on first run |

## Quick Start

```bash
git clone https://github.com/Tanishka-01/tada-clinical-eval
cd tada-clinical-eval
pip install -r requirements.txt
cp .env.example .env
# Add your HUME_API_KEY to .env (optional — ECS metric only)
python assets/create_reference_audio.py  # Generate reference audio
python scripts/run_quick.py              # 20 sentences, ~10 minutes on CPU
python scripts/run_eval.py               # Full 200-sentence evaluation
```

## Standalone Report Generation

```bash
python scripts/generate_report.py \
  --results-json results/results_20260314_120000.json \
  --regenerate-figures \
  --report-path tada_clinical_eval_report.pdf
```

## Running Tests

```bash
pytest tests/                     # All tests
pytest tests/ -m "not slow"       # Skip model-loading tests
pytest tests/test_dataset.py -v   # Dataset validation
pytest tests/test_metrics.py -v   # Metrics unit tests
pytest tests/test_report.py -v    # Report generation tests
```

## Project Structure

```
tada-clinical-eval/
├── dataset/          # 200 clinical sentences with annotations
├── models/           # TADA, edge-tts, Coqui wrappers
├── metrics/          # CER, RTF, Speaker Similarity, ECS, CTA, PAS
├── evaluation/       # Runner, storage, validation
├── analysis/         # Statistics, charts, tables
├── report/           # PDF report generator (reportlab)
├── tests/            # Full test suite
├── assets/           # Reference audio
└── scripts/          # Entry points
```

## Addressing TADA GitHub Issue #9

This harness empirically answers: *"Why choose CER > 0.15 as the hallucination threshold?"*

Our CTA analysis shows that CER > 0.15 is too permissive for clinical contexts. A model with CER = 0.08 overall can still hallucinate specific medication names at rates that would be clinically unacceptable. We recommend reporting CTA separately from overall CER, with a CTA threshold of > 0.95 for healthcare deployment.

## Results

*Run `python scripts/run_eval.py` to populate this section.*

## Citation

```bibtex
@misc{wani2026tadaclinical,
  title={Evaluating TADA for Clinical Voice AI: A Domain-Specific Benchmark for Therapeutic Speech Generation},
  author={Wani, Tanishka},
  year={2026},
  url={https://github.com/Tanishka-01/tada-clinical-eval}
}
```

```bibtex
@article{dang2026tada,
  title={TADA: A Generative Framework for Speech Modeling via Text-Acoustic Dual Alignment},
  author={Dang, Trung and Rao, Sharath and Gupta, Ananya and Gagne, Christopher and
          Tzirakis, Panagiotis and Baird, Alice and Clapa, Jakub Piotr and Chin, Peter
          and Cowen, Alan},
  journal={arXiv preprint arXiv:2602.23068},
  year={2026}
}
```

## Safety Note

This harness is for research purposes only. Do not use TTS systems evaluated here for actual clinical decision-making without professional review and regulatory compliance assessment.

## License

MIT License
