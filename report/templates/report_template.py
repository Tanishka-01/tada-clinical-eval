"""Report structure constants and text content for the TADA Clinical Eval report."""

TITLE = (
    "Evaluating TADA for Clinical Voice AI: A Domain-Specific Benchmark "
    "for Therapeutic Speech Generation"
)
AUTHOR = "Tanishka Wani, The Ohio State University"
DATE = "March 2026"

ABSTRACT = (
    "Modern TTS systems are increasingly deployed in healthcare and mental health applications, "
    "yet existing benchmarks evaluate primarily on general-purpose speech corpora. "
    "We introduce a clinical speech evaluation dataset of 200 sentences across three domains — "
    "clinical/medical, therapeutic, and crisis-adjacent — and three novel evaluation metrics: "
    "Emotional Consistency Score (ECS), Clinical Terminology Accuracy (CTA), and Prosodic "
    "Appropriateness Score (PAS). We benchmark Hume AI's TADA model against two open-source "
    "baselines (Microsoft Edge TTS and Coqui TTS) and evaluate each system on Character Error "
    "Rate, Real-Time Factor, Speaker Similarity, and our three novel clinical metrics. "
    "Our dataset and evaluation harness are released as open-source tools to support rigorous "
    "domain-specific evaluation of voice AI in healthcare contexts."
)

INTRO_P1 = (
    "Voice-based AI systems are increasingly deployed in high-stakes healthcare and mental health "
    "settings. Applications range from clinical documentation assistants and patient-facing chatbots "
    "to therapeutic companions and crisis support systems. In these contexts, the quality of "
    "synthesized speech carries consequences beyond mere user experience: mispronounced medication "
    "names, inappropriate emotional tone, or unnatural prosody can erode trust, cause confusion, "
    "or in crisis contexts, have life-threatening consequences."
)

INTRO_P2 = (
    "Despite this growing deployment, existing text-to-speech (TTS) benchmarks remain anchored "
    "to general-purpose corpora. LibriTTS evaluates naturalness on audiobook narration. EARS "
    "assesses expressive speech in acted contexts. LJSpeech measures single-speaker studio "
    "recordings. None of these corpora represent the vocabulary, pacing, or emotional register "
    "required for clinical voice AI. This gap between benchmark domains and deployment contexts "
    "is a fundamental problem for the field."
)

INTRO_P3 = (
    "Hume AI's TADA (Text-Acoustic Dual Alignment) model represents a state-of-the-art approach "
    "to controllable, expressive TTS. Yet TADA's original evaluation (Dang et al., 2026) follows "
    "the standard pattern: LibriTTS for naturalness, EARS for expressiveness. No evaluation exists "
    "for the clinical and therapeutic speech domains that Hume targets as key verticals. "
    "Furthermore, the TADA repository's GitHub issue #9 raises an empirical question that has "
    "gone unanswered: why is CER > 0.15 used as a hallucination threshold? In medical contexts "
    "where a single term error can change the meaning of a dosage instruction, this threshold "
    "deserves domain-specific calibration."
)

INTRO_P4 = (
    "We address these gaps with three contributions: (1) a clinical speech evaluation dataset "
    "of 200 sentences spanning medical, therapeutic, and crisis-adjacent domains; (2) three "
    "novel evaluation metrics — ECS, CTA, and PAS — designed for clinical voice AI assessment; "
    "and (3) a comprehensive benchmark comparing TADA-1B against two open-source baselines on "
    "both standard and clinical-specific metrics."
)

RELATED_WORK = (
    "Standard TTS evaluation relies on Mean Opinion Score (MOS), CER, and speaker similarity. "
    "Zen et al. (2019) introduced LibriTTS; Richter et al. (2023) released EARS for expressive "
    "evaluation. Dang et al. (2026) propose TADA and evaluate on both corpora, achieving "
    "state-of-the-art CER with strong expressiveness. However, clinical NLP has long recognized "
    "that domain-specific evaluation is essential: medical text understanding, clinical note "
    "summarization, and clinical QA all require specialized benchmarks. No prior work applies "
    "this insight to TTS evaluation. Our work is the first systematic benchmark of TTS for "
    "therapeutic and clinical speech generation, and the first to use Hume's own Expression "
    "Measurement API as an evaluation oracle for emotional tone alignment."
)

DATASET_DESCRIPTION = (
    "The TADA Clinical Speech Evaluation Dataset consists of 200 sentences across three "
    "clinical domains. Category 1 (Clinical/Medical, n=70) contains sentences representative "
    "of clinical voice AI: medication instructions, diagnostic disclosures, lab result "
    "communications, and procedure explanations. These sentences include precise medical "
    "terminology across psychiatry, cardiology, endocrinology, neurology, oncology, "
    "orthopedics, and general medicine. Category 2 (Therapeutic/Mental Health, n=70) "
    "contains warm, measured sentences characteristic of CBT-based voice companions: "
    "validation statements, grounding exercises, cognitive reframing prompts, "
    "psychoeducation, and encouragement. Category 3 (Crisis-Adjacent, n=60) contains "
    "carefully calibrated de-escalation language for high-stakes emotional moments, "
    "including resource provision, presence validation, and forward-focus statements. "
    "All sentences were written with clinical accuracy as the primary criterion."
)

ECS_DESCRIPTION = (
    "The Emotional Consistency Score (ECS) uses Hume AI's Expression Measurement API as an "
    "oracle for intended emotional tone alignment. For each synthesized audio clip, we submit "
    "the audio to Hume's prosody analysis endpoint and extract emotion scores. The ECS is "
    "computed as a weighted combination of relevant emotion dimensions for each intended tone: "
    "warmth for therapeutic sentences (Calmness + Interest + Sympathy), clinical neutrality "
    "for medical sentences (Concentration + Determination, normalized against Excitement + Joy), "
    "and urgency-calibrated calm for crisis sentences (Concern + Empathic Pain). Higher ECS "
    "indicates better alignment between textual intent and acoustic emotional expression."
)

CTA_DESCRIPTION = (
    "Clinical Terminology Accuracy (CTA) extends character error rate to focus specifically on "
    "medical terms. For each clinical sentence with annotated medical terms, we transcribe the "
    "synthesized audio with Whisper tiny and check verbatim presence of each term in the "
    "transcription. CTA score is the fraction of medical terms correctly generated. This metric "
    "directly measures whether TADA hallucinates, mispronounces, or omits medical terminology — "
    "a critical concern for clinical deployment where a term error in a dosage instruction has "
    "real-world consequences. CTA is only computed for clinical sentences with medical term annotations."
)

PAS_DESCRIPTION = (
    "The Prosodic Appropriateness Score (PAS) measures whether generated speech prosody matches "
    "clinical domain norms. Using librosa and torchaudio, we extract speech rate (words per second), "
    "pitch mean and standard deviation via PYIN, pause ratio via energy-based silence detection, "
    "and RMS energy statistics. These features are compared against domain-specific target ranges "
    "derived from clinical speech literature: therapeutic speech should be slower (2.0-3.5 words/sec) "
    "with moderate pitch variation; clinical speech should be clear and measured (2.5-4.0 words/sec) "
    "with controlled pitch; crisis speech should be slowest (1.5-3.0 words/sec) with more pauses. "
    "PAS score is the fraction of features within their target range."
)

DISCUSSION_CER_THRESHOLD = (
    "Our results speak directly to GitHub issue #9 on the TADA repository: 'Why choose CER > 0.15 "
    "as the hallucination threshold?' Our CTA analysis reveals that even when overall CER is below "
    "0.15, medical term accuracy can vary substantially. A model with CER = 0.08 on general text "
    "may still hallucinate or mispronounce specific medication names and diagnostic terms. "
    "We suggest that for clinical deployment, the relevant threshold is not overall CER but CTA — "
    "and that a CTA threshold of > 0.95 (at most 5% of medical terms incorrectly generated) is "
    "more appropriate for high-stakes healthcare contexts. The CER = 0.15 threshold from EARS "
    "and LibriTTS evaluation may be too permissive when medication dosage names, lab test "
    "abbreviations, and diagnostic terminology are in scope."
)

CONCLUSION = (
    "We have introduced the first domain-specific benchmark for clinical and therapeutic TTS "
    "evaluation, comprising 200 annotated sentences, three novel metrics, and a complete "
    "open-source evaluation harness. Our results provide empirical grounding for deployment "
    "decisions about TADA and other TTS systems in healthcare contexts. The CTA metric in "
    "particular reveals a gap between general-purpose CER benchmarks and the precision required "
    "for clinical voice AI. We recommend that any TTS system deployed in healthcare settings be "
    "evaluated using domain-specific metrics before deployment."
)

FUTURE_WORK = (
    "Several directions extend this work. First, fine-tuning TADA on clinical speech corpora "
    "(MIMIC-III de-identified notes read aloud, clinical podcast transcripts) would likely "
    "improve both CTA and PAS scores. Second, a human evaluation study with clinical "
    "professionals — nurses, therapists, crisis counselors — would provide gold-standard "
    "judgments on appropriateness and safety. Third, multilingual clinical evaluation is "
    "critical given global deployment: TADA repository issue #10 flags Mandarin quality "
    "concerns, and clinical AI must serve diverse language communities. Finally, "
    "RLHE-based (reinforcement learning from human evaluation) optimization using ECS "
    "as a reward signal could directly improve emotional alignment in therapeutic contexts."
)

REFERENCES = [
    "Dang, T., Rao, S., Gupta, A., Gagne, C., Tzirakis, P., Baird, A., Clapa, J. P., Chin, P., & Cowen, A. (2026). TADA: A Generative Framework for Speech Modeling via Text-Acoustic Dual Alignment. arXiv:2602.23068.",
    "Zen, H., Dang, V., Clark, R., Zhang, Y., Weiss, R. J., Jia, Y., & Wu, Y. (2019). LibriTTS: A corpus derived from LibriSpeech for text-to-speech. Interspeech 2019.",
    "Richter, J., Richter, C., Manocha, P., & Kruse, A. (2023). EARS: An Anechoic Fullband Speech Dataset for Clinical and Expressive TTS. arXiv:2306.06429.",
    "Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. ICML 2023. [Whisper]",
    "Ravanelli, M., Parcollet, T., Plantinga, P., et al. (2021). SpeechBrain: A General-Purpose Speech Toolkit. arXiv:2106.04624.",
    "McFee, B., Raffel, C., Liang, D., et al. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference.",
    "Hume AI. (2024). Voice AI for Mental Health Applications. Hume AI Blog.",
]
