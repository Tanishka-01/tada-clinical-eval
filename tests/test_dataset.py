import pytest
from dataset.sentences import SENTENCES, get_sentences_by_category


def test_total_sentence_count():
    assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"


def test_category_counts():
    clinical = get_sentences_by_category("clinical")
    therapeutic = get_sentences_by_category("therapeutic")
    crisis = get_sentences_by_category("crisis")
    assert len(clinical) == 70, f"Expected 70 clinical, got {len(clinical)}"
    assert len(therapeutic) == 70, f"Expected 70 therapeutic, got {len(therapeutic)}"
    assert len(crisis) == 60, f"Expected 60 crisis, got {len(crisis)}"


def test_sentence_has_required_fields():
    required = [
        "id", "text", "category", "subcategory",
        "contains_medical_terms", "medical_terms",
        "intended_tone", "expected_speech_rate",
    ]
    for s in SENTENCES:
        for field in required:
            assert field in s, f"Sentence {s.get('id', '?')} missing field '{field}'"


def test_medical_terms_populated_for_clinical():
    for s in get_sentences_by_category("clinical"):
        if s["contains_medical_terms"]:
            assert len(s["medical_terms"]) > 0, (
                f"{s['id']} has contains_medical_terms=True but empty medical_terms"
            )


def test_ids_are_unique():
    ids = [s["id"] for s in SENTENCES]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_no_empty_sentences():
    for s in SENTENCES:
        assert s["text"].strip(), f"Empty text in sentence {s['id']}"


def test_valid_categories():
    valid = {"clinical", "therapeutic", "crisis"}
    for s in SENTENCES:
        assert s["category"] in valid, f"Invalid category in {s['id']}: {s['category']}"


def test_valid_intended_tones():
    valid = {"neutral", "warm", "urgent", "calm"}
    for s in SENTENCES:
        assert s["intended_tone"] in valid, (
            f"Invalid intended_tone in {s['id']}: {s['intended_tone']}"
        )


def test_valid_speech_rates():
    valid = {"normal", "slow", "measured"}
    for s in SENTENCES:
        assert s["expected_speech_rate"] in valid, (
            f"Invalid expected_speech_rate in {s['id']}: {s['expected_speech_rate']}"
        )
